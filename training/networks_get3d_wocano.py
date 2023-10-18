# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import persistence
import nvdiffrast.torch as dr
# from training.sample_camera_distribution import sample_camera, create_camera_from_angle
from uni_rep.rep_3d.dmtet import DMTetGeometry
from uni_rep.camera.perspective_camera import PerspectiveCamera
from uni_rep.render.neural_render import NeuralRender
from training.discriminator_architecture import Discriminator
from training.geometry_predictor import MappingNetwork, ToRGBLayer, \
    TriPlaneTexGeo, SuperresolutionHybrid4X, SuperresolutionHybrid8X

from pytorch3d.ops.knn import knn_points

from smplx import create
from training.smpl_utils import get_canonical_pose, face_vertices, cal_sdf_batch, \
    batch_index_select, batch_transform, batch_transform_normal, get_eikonal_term


@persistence.persistent_class
class DMTETSynthesisNetwork(torch.nn.Module):
    def __init__(
            self,
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output image resolution.
            img_channels,  # Number of color channels.
            device='cuda',
            data_camera_mode='carla',
            geometry_type='normal',
            tet_res=64,  # Resolution for tetrahedron grid
            render_type='neural_render',  # neural type
            use_tri_plane=False,
            n_views=1,
            tri_plane_resolution=128,
            deformation_multiplier=2.0,
            feat_channel=128,
            mlp_latent_channel=256,
            dmtet_scale=1.8, # adjust the scale according to the canonical space or observation space?
            inference_noise_mode='random',
            one_3d_generator=False,
            **block_kwargs,  # Arguments for SynthesisBlock.
    ):  #
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.device = device
        self.one_3d_generator = one_3d_generator
        self.inference_noise_mode = inference_noise_mode
        self.dmtet_scale = dmtet_scale
        self.deformation_multiplier = deformation_multiplier
        self.geometry_type = geometry_type

        self.data_camera_mode = data_camera_mode
        self.n_freq_posenc_geo = 1
        self.render_type = render_type
        
        # thuman: tensor([-0.8394, -1.2347, -0.7895]) tensor([0.8431, 0.9587, 0.9094])
        # render_people: tensor([-1.0043, -1.3278, -0.4914]) tensor([1.0058, 0.8841, 0.7573])
        self.obs_bbox_y_max = 0.96
        self.obs_bbox_y_min = -1.33
        self.dmtet_scale = self.obs_bbox_y_max - self.obs_bbox_y_min
        self.obs_bbox_y_center = 0.5 * (self.obs_bbox_y_max + self.obs_bbox_y_min)

        # self.cano_bbox_y_max = 0.65
        # self.cano_bbox_y_min = -1.31
        # self.cano_bbox_y_center = 0.5 * (self.cano_bbox_y_max + self.cano_bbox_y_min)
        self.cano_bbox_length = 2.0 # range: [-1.185, 0.815]
        # 1.0: [[-0.8719, -1.0534, -0.1224], [0.8731, 0.5554, 0.1691]]
        # 1.1: [[-1.0464, -1.2143, -0.1516], [1.0476, 0.7163, 0.1983]]  
        
        # dim_embed_geo = 3 * self.n_freq_posenc_geo * 2
        self.w_dim = w_dim
        self.c_cam_dim = 16
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.n_views = n_views
        self.grid_res = tet_res

        # Camera defination, we follow the defination from Blender (check the render_shapenet_data/rener_shapenet.py for more details)
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=self.device)

        # Renderer we used.
        dmtet_renderer = NeuralRender(device, camera_model=dmtet_camera)

        cano_bbox = torch.tensor([
            [-1.175, -1.500, -1.175], 
            [ 1.175,  0.850,  1.175]]).float().to(self.device)
        
        self.register_buffer('cano_bbox', cano_bbox)

        # Geometry class for DMTet
        self.dmtet_geometry = DMTetGeometry(
            grid_res=self.grid_res, scale=self.dmtet_scale, renderer=dmtet_renderer, render_type=render_type,
            device=self.device)

        self.feat_channel = feat_channel
        self.mlp_latent_channel = mlp_latent_channel
        self.use_tri_plane = use_tri_plane
        if self.one_3d_generator:
            # Use a unified generator for both geometry and texture generation
            # shape_min, shape_max = self.dmtet_geometry.getAABB()
            # shape_min = shape_min.min()
            # shaape_lenght = shape_max.max() - shape_min
            shape_min = -0.5 * self.cano_bbox_length
            shaape_lenght = self.cano_bbox_length
            self.generator = TriPlaneTexGeo(
                w_dim=w_dim,
                img_channels=self.feat_channel,
                shape_min=shape_min,
                shape_lenght=shaape_lenght,
                cano_bbox=self.cano_bbox,
                tri_plane_resolution=tri_plane_resolution,
                device=self.device,
                mlp_latent_channel=self.mlp_latent_channel,
                **block_kwargs)
        else:
            raise NotImplementedError

        self.channels_last = False
        if self.feat_channel > 3:
            # Final layer to convert the texture field to RGB color, this is only a fully connected layer.
            self.to_rgb = ToRGBLayer(
                self.feat_channel, self.img_channels, w_dim=w_dim,
                conv_clamp=256, channels_last=self.channels_last, device=self.device)
            # self.rgb_decoder = OSGDecoder(self.feat_channel, device=device)

        if self.img_resolution == 256:
            self.superresolution = SuperresolutionHybrid4X(
                channels=self.feat_channel, 
                img_resolution=self.img_resolution, 
                sr_num_fp16_res=4, 
                sr_antialias=True, 
                channel_base=32768, 
                channel_max=512, 
                fused_modconv_default='inference_only')
        elif self.img_resolution == 512:
            self.superresolution = SuperresolutionHybrid8X(
                channels=self.feat_channel, 
                img_resolution=self.img_resolution, 
                sr_num_fp16_res=4, 
                sr_antialias=True, 
                channel_base=32768, 
                channel_max=512, 
                fused_modconv_default='inference_only')
        else:
            raise NotImplementedError

        self.glctx = None

         # Define SMPL BODY MODEL
        self.body_model = create(model_path='./smplx/models', model_type='smpl', gender='neutral').to(self.device)
        # Define X-Pose
        pose_canonical = torch.from_numpy(np.array(get_canonical_pose())).float().to(self.device)
        lbs_weights = self.body_model.lbs_weights.to(self.device)
        faces = torch.from_numpy(self.body_model.faces.astype(np.int64)).to(self.device)
        zero_smpl_beta = torch.tensor([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        ]).float().to(self.device)
        
        # # 1.0
        # canonical_bbox = torch.tensor([[-0.8719, -1.0534, -0.1224], [0.8731, 0.5554, 0.1691]]).reshape(1,2,3).float().to(self.device)
        # # 1.1
        # canonical_bbox = torch.tensor([[-1.0464, -1.2143, -0.1516], [1.0476, 0.7163, 0.1983]]).reshape(1,2,3).float().to(self.device)
        # # 1.2
        # canonical_bbox = torch.tensor([[-1.2209, -1.3752, -0.1807], [1.2221, 0.8772, 0.2274]]).reshape(1,2,3).float().to(self.device)
        # final
        canonical_bbox = torch.tensor([[-1.05, -1.20, -0.18], [1.05, 0.70, 0.22]]).reshape(1,2,3).float().to(self.device)

        self.register_buffer('pose_canonical', pose_canonical)
        self.register_buffer('lbs_weights', lbs_weights)
        self.register_buffer('faces', faces)
        self.register_buffer('zero_smpl_beta', zero_smpl_beta)
        self.register_buffer('canonical_bbox', canonical_bbox)

        self.canonical_smpl_info = self.get_canonical_smpl_info()

    def transform_points(self, p, for_geo=False):
        pi = np.pi
        assert for_geo
        L = self.n_freq_posenc_geo
        p_transformed = torch.cat(
            [torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def get_canonical_smpl_info(self):
        with torch.no_grad():
            # Canonical Space
            body_model_params_canonical = {'body_pose': self.pose_canonical.unsqueeze(0).repeat(1, 1), 
            'betas': self.zero_smpl_beta.unsqueeze(0).repeat(1, 1)}

            body_model_out_canonical = self.body_model(**body_model_params_canonical, return_verts=True)
            verts_canonical = body_model_out_canonical['vertices']
            triangles_canonical = face_vertices(verts_canonical, self.faces, self.device)
            verts_transform_canonical = body_model_out_canonical['vertices_transform']
            shape_offsets_canonical = body_model_out_canonical['shape_offsets']
            pose_offsets_canonical = body_model_out_canonical['pose_offsets']
            del body_model_out_canonical

            canonical_smpl_kwargs = {
                'verts_canonical': verts_canonical,
                'triangles_canonical': triangles_canonical,
                'verts_transform_canonical':  verts_transform_canonical,
                'shape_offsets_canonical': shape_offsets_canonical,
                'pose_offsets_canonical': pose_offsets_canonical,
                # 'faces': self.faces,
                'lbs_weights': self.lbs_weights,
            }
            return canonical_smpl_kwargs


    def calc_ober2cano_transform(
        self, verts_transform, verts_transform_canonical, 
        shape_offsets, shape_offsets_canonical,
        pose_offsets, pose_offsets_canonical):

        ober2cano_transform = torch.inverse(verts_transform).clone()
        ober2cano_transform[..., :3, 3] = ober2cano_transform[..., :3, 3] + (shape_offsets_canonical - shape_offsets)
        ober2cano_transform[..., :3, 3] = ober2cano_transform[..., :3, 3] + (pose_offsets_canonical - pose_offsets)
        ober2cano_transform = torch.matmul(verts_transform_canonical, ober2cano_transform)
        return ober2cano_transform
    
    # def inverse_skinning(self, coords, verts, triangles_obs, ober2cano_transform):
    #     batch_size = coords.shape[0]
    #     batch_obs_body_sdf = cal_sdf_batch(verts, self.faces,
    #                 triangles_obs, coords, self.device) # [batch_size, 98653, 1]
    #     batch_mask = (batch_obs_body_sdf < 0.3).squeeze(-1) # [batch_size, 98653]

    #     for idx in range(batch_size):
    #         cur_mask = batch_mask[idx:idx+1]
    #         cur_coords = coords[idx:idx+1]
    #         valid_coords_obs = cur_coords[cur_mask].unsqueeze(0)
    #         valid_coords_cano, valid = self.unpose(valid_coords_obs, verts[idx:idx+1], ober2cano_transform[idx:idx+1])
    #         coords[idx:idx+1][cur_mask] = valid_coords_cano

    #     return coords, batch_obs_body_sdf, batch_mask

    def canonical_mapping(self, coords, verts, body_bbox, ober2cano_transform):
        mask, filtered_coords_mask, max_length, filtered_coords, _ = self.filter_and_pad(coords, body_bbox)
        filtered_coords_ori = filtered_coords.clone()
        filtered_coords_cano, _ = self.unpose(filtered_coords, verts, ober2cano_transform)

        filtered_coords = filtered_coords_cano

        canonical_mask, canonical_filtered_coords_mask, max_length, filtered_coords, filtered_coords_ori = \
                self.filter_and_pad(filtered_coords, self.canonical_bbox.repeat(coords.shape[0], 1, 1))
        
         # Combine two masks (make clone of the masks to avoid inplace replacement)
        new_mask = mask.clone()
        new_mask[mask] = canonical_mask[filtered_coords_mask]
        mask = new_mask

        new_canonical_mask = canonical_filtered_coords_mask.clone()
        new_canonical_mask[canonical_filtered_coords_mask] = filtered_coords_mask[canonical_mask]
        canonical_filtered_coords_mask = new_canonical_mask

        return filtered_coords, filtered_coords_ori, mask, canonical_filtered_coords_mask, max_length

    def canonical_mapping_normal(self, coords, verts, body_bbox, ober2cano_transform):
        mask, filtered_coords_mask, max_length, filtered_coords, _ = self.filter_and_pad(coords, body_bbox)
        #filtered_coords_ori = filtered_coords.clone()
        filtered_coords_cano, rotate_matrix = self.unpose_normal(filtered_coords, verts, ober2cano_transform)

        filtered_coords = filtered_coords_cano

        canonical_mask, canonical_filtered_coords_mask, max_length, filtered_coords, filtered_rotate_matrix = \
                self.filter_and_pad_normal(filtered_coords, self.canonical_bbox.repeat(coords.shape[0], 1, 1), rotate_matrix)
        
         # Combine two masks (make clone of the masks to avoid inplace replacement)
        new_mask = mask.clone()
        new_mask[mask] = canonical_mask[filtered_coords_mask]
        mask = new_mask

        new_canonical_mask = canonical_filtered_coords_mask.clone()
        new_canonical_mask[canonical_filtered_coords_mask] = filtered_coords_mask[canonical_mask]
        canonical_filtered_coords_mask = new_canonical_mask

        return filtered_coords, filtered_rotate_matrix, mask, canonical_filtered_coords_mask, max_length

    @staticmethod
    def filter_and_pad(coords, bbox, coords_ori=None, bbox_ori=None):
        # filter out coords that are out of the bbox and pad the batch to same length
        device = coords.device
        batch_size, num_pts, _ = coords.shape
        boxed_coords = 2 * (coords-bbox[:, 0:1]) / (bbox[:, 1:2]-bbox[:, 0:1]) - 1
        mask = boxed_coords.abs().amax(-1) <= 1
        max_length = torch.max(torch.sum(mask, 1))
        filtered_coords_mask = torch.arange(max_length, device=device).unsqueeze(0).repeat(batch_size, 1) < torch.sum(mask, 1, keepdim=True)
        filtered_coords = bbox[:, 0:1].repeat(1, max_length, 1)
        filtered_coords[filtered_coords_mask] = coords[mask]
        filtered_coords_ori = None

        if coords_ori is not None and bbox_ori is not None:
            filtered_coords_ori = bbox_ori[:, 0:1].repeat(1, max_length, 1)
            filtered_coords_ori[filtered_coords_mask] = coords_ori[mask]

        return mask, filtered_coords_mask, max_length, filtered_coords, filtered_coords_ori
    
    @staticmethod
    def filter_and_pad_normal(coords, bbox, rotate_matrix):
        # filter out coords that are out of the bbox and pad the batch to same length
        device = coords.device
        batch_size, num_pts, _ = coords.shape
        boxed_coords = 2 * (coords-bbox[:, 0:1]) / (bbox[:, 1:2]-bbox[:, 0:1]) - 1
        mask = boxed_coords.abs().amax(-1) <= 1
        max_length = torch.max(torch.sum(mask, 1))
        filtered_coords_mask = torch.arange(max_length, device=device).unsqueeze(0).repeat(batch_size, 1) < torch.sum(mask, 1, keepdim=True)
        filtered_coords = bbox[:, 0:1].repeat(1, max_length, 1) # (batch_size, max_length, 3)
        filtered_coords[filtered_coords_mask] = coords[mask]

        filtered_rotate_matrix = torch.eye(3).type_as(rotate_matrix).view(1, 1, 3, 3).repeat(batch_size, max_length, 1, 1)
        filtered_rotate_matrix[filtered_coords_mask] = rotate_matrix[mask]

        return mask, filtered_coords_mask, max_length, filtered_coords, filtered_rotate_matrix

    def unpose(self, coords, verts, ober2cano_transform, dis_threshold=0.1):
        bs, nv = coords.shape[:2]
        coords_dist, coords_transform_inv = self.get_neighbs(
            coords, verts, ober2cano_transform.clone())
        coords_valid = torch.lt(coords_dist, dis_threshold).float()
        coords_unposed = batch_transform(coords_transform_inv, coords)
        return coords_unposed, coords_valid
    
    def unpose_normal(self, coords, verts, ober2cano_transform, dis_threshold=0.1):
        bs, nv = coords.shape[:2]
        coords_dist, coords_transform_inv = self.get_neighbs(
            coords, verts, ober2cano_transform.clone())
        coords_valid = torch.lt(coords_dist, dis_threshold).float()
        coords_unposed, skinning_rotate_transform = batch_transform_normal(coords_transform_inv, coords)
        return coords_unposed, skinning_rotate_transform

    def get_neighbs(self, coords, verts, verts_transform_inv, lbs_weights=None, k_neigh=1, weight_std=0.1):
        bs, nv = verts.shape[:2]

        with torch.no_grad():
            # try:
            neighbs_dist, neighbs, _ = knn_points(coords, verts, K=k_neigh)
            neighbs_dist = torch.sqrt(neighbs_dist)

        # weight_std2 = 2. * weight_std ** 2
        # coords_neighbs_lbs_weight = lbs_weights[neighbs] # (bs, n_rays*K, k_neigh, 24)
        # # (bs, n_rays*K, k_neigh)
        # coords_neighbs_weight_conf = torch.exp(-torch.sum(torch.abs(coords_neighbs_lbs_weight - coords_neighbs_lbs_weight[..., 0:1, :]), dim=-1) / weight_std2)
        # coords_neighbs_weight_conf = torch.gt(coords_neighbs_weight_conf, 0.9).float()  # why 0.9?
        # coords_neighbs_weight = torch.exp(-neighbs_dist) # (bs, n_rays*K, k_neigh)
        # coords_neighbs_weight = coords_neighbs_weight * coords_neighbs_weight_conf
        # coords_neighbs_weight = coords_neighbs_weight / coords_neighbs_weight.sum(-1, keepdim=True) # (bs, n_rays*K, k_neigh)

        # coords_neighbs_transform_inv = batch_index_select(verts_transform_inv, neighbs, self.device) # (bs, n_rays*K, k_neigh, 4, 4)
        # coords_transform_inv = torch.sum(coords_neighbs_weight.unsqueeze(-1).unsqueeze(-1)*coords_neighbs_transform_inv, dim=2) # (bs, n_rays*K, 4, 4)
        # coords_dist = torch.sum(coords_neighbs_weight*neighbs_dist, dim=2, keepdim=True) # (bs, n_rays*K, 1)

        coords_neighbs_transform_inv = batch_index_select(verts_transform_inv, neighbs, self.device) # (bs, n_rays*K, k_neigh, 4, 4)
        coords_transform_inv = coords_neighbs_transform_inv.squeeze(2)

        return neighbs_dist, coords_transform_inv
        # return coords_dist, coords_transform_inv

    def get_sdf_deformation_prediction(
            self, ws, position=None, sdf_feature=None,
            canonical_mapping_kwargs=None
            ):
        '''
        Predict SDF and deformation for tetrahedron vertices
        :param ws: latent code for the geometry
        :param position: the location of tetrahedron vertices
        :param sdf_feature: triplane feature map for the geometry
        :return:
        '''
        batch_size = ws.shape[0]
        if position is None:
            init_position = self.dmtet_geometry.verts.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        else:
            init_position = position

        num_pts = init_position.shape[1]
        # Step 1: predict the SDF and deformation
        if self.one_3d_generator:
            """
            init_position[:, :, 1] += self.obs_bbox_y_center
            # cano_position, batch_obs_body_sdf, batch_mask = self.inverse_skinning(init_position, **canonical_mapping_kwargs)
            sample_coordinates_cano, sample_coordinates_obs, mask, coordinates_mask, max_length = self.canonical_mapping(init_position, **canonical_mapping_kwargs)
            batch_cano_body_sdf = cal_sdf_batch(
                self.canonical_smpl_info['verts_canonical'].repeat(batch_size, 1, 1), 
                self.faces,
                self.canonical_smpl_info['triangles_canonical'].repeat(batch_size, 1, 1, 1), 
                sample_coordinates_cano, 
                self.device) # [batch_size, 98653, 1]
            sample_coordinates_cano[:, :, 1] -= self.obs_bbox_y_center
            """

            mask, coordinates_mask, max_length, sample_coordinates_obs, _ = self.filter_and_pad(init_position, self.cano_bbox[None].repeat(init_position.shape[0], 1, 1))
            batch_body_sdf = cal_sdf_batch(
                canonical_mapping_kwargs['verts'], 
                self.faces,
                canonical_mapping_kwargs['triangles_obs'], 
                sample_coordinates_obs, 
                self.device)
            
            sdf_cano, deformation_cano = self.generator.get_sdf_def_prediction(
                sdf_feature, ws_geo=ws,
                position=sample_coordinates_obs)
            
            sdf_cano = sdf_cano + batch_body_sdf
            
            sdf = torch.zeros((batch_size, num_pts, 1)).to(self.device) + 10.0
            deformation = torch.zeros((batch_size, num_pts, 3)).to(self.device)

            sdf[mask] = sdf_cano[coordinates_mask]
            deformation[mask] = deformation_cano[coordinates_mask]
        else:
            raise NotImplementedError

        # Step 2: Normalize the deformation to avoid the flipped triangles.
        deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation)

        sdf_reg_loss = torch.zeros(sdf.shape[0], device=sdf.device, dtype=torch.float32)

        # Step 3: Fix some sdf if we observe empty shape (full positive or full negative)
        pos_shape = torch.sum((sdf.squeeze(dim=-1) > 0).int(), dim=-1)
        neg_shape = torch.sum((sdf.squeeze(dim=-1) < 0).int(), dim=-1)
        zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        if torch.sum(zero_surface).item() > 0:
            update_sdf = torch.zeros_like(sdf[0:1])
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            update_sdf[:, self.dmtet_geometry.center_indices] += (1.0 - min_sdf)  # greater than zero
            update_sdf[:, self.dmtet_geometry.boundary_indices] += (-1 - max_sdf)  # smaller than zero
            new_sdf = torch.zeros_like(sdf)
            for i_batch in range(zero_surface.shape[0]):
                if zero_surface[i_batch]:
                    new_sdf[i_batch:i_batch + 1] += update_sdf
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
            sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)

        # Step 4: Here we remove the gradient for the bad sdf (full positive or full negative)
        final_sdf = []
        final_def = []
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                final_sdf.append(sdf[i_batch: i_batch + 1].detach())
                final_def.append(deformation[i_batch: i_batch + 1].detach())
            else:
                final_sdf.append(sdf[i_batch: i_batch + 1])
                final_def.append(deformation[i_batch: i_batch + 1])
        sdf = torch.cat(final_sdf, dim=0)
        deformation = torch.cat(final_def, dim=0)
        return sdf, deformation, sdf_reg_loss

    def get_geometry_prediction(self, ws, sdf_feature=None, canonical_mapping_kwargs=None):
        '''
        Function to generate mesh with give latent code
        :param ws: latent code for geometry generation
        :param sdf_feature: triplane feature for geometry generation
        :return:
        '''
        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        sdf, deformation, sdf_reg_loss = self.get_sdf_deformation_prediction(
            ws,
            sdf_feature=sdf_feature,
            canonical_mapping_kwargs=canonical_mapping_kwargs)
        v_deformed = self.dmtet_geometry.verts.unsqueeze(dim=0).expand(sdf.shape[0], -1, -1) + deformation
        tets = self.dmtet_geometry.indices
        n_batch = ws.shape[0]
        v_list = []
        f_list = []

        # Step 2: Using marching tet to obtain the mesh
        for i_batch in range(n_batch):
            verts, faces = self.dmtet_geometry.get_mesh(
                v_deformed[i_batch], sdf[i_batch].squeeze(dim=-1),
                with_uv=False, indices=tets)
            # verts[:, 1] += self.obs_bbox_y_center
            v_list.append(verts)
            f_list.append(faces)

        return v_list, f_list, sdf, deformation, v_deformed, sdf_reg_loss


    def get_texture_prediction(self, ws, tex_pos, hard_mask, tex_feature, canonical_mapping_kwargs):
        '''
        Predict Texture given latent codes
        :param ws: Latent code for texture generation
        :param tex_pos: Position we want to query the texture field
        :param ws_geo: latent code for geometry
        :param hard_mask: 2D silhoueete of the rendered image
        :param tex_feature: the triplane feature map
        :return:
        '''
        tex_pos = torch.cat(tex_pos, dim=0)
        
        tex_pos = tex_pos * hard_mask.float()
        batch_size = tex_pos.shape[0]
        tex_pos = tex_pos.reshape(batch_size, -1, 3)   # [batch_size, res*res, 3]

        ###################
        # We use mask to get the texture location (to save the memory)
        # if hard_mask is not None:
        n_point_list = torch.sum(hard_mask.long().reshape(hard_mask.shape[0], -1), dim=-1) # [batch_size]
        sample_tex_pose_list = []
        max_point = n_point_list.max() # max_length among batch_size number
        expanded_hard_mask = hard_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3) > 0.5 # [batch_size, res*res, 3]
        for i in range(tex_pos.shape[0]):
            tex_pos_one_shape = tex_pos[i][expanded_hard_mask[i]].reshape(1, -1, 3) # [batch_size, n_point_list[i], 3]

            if tex_pos_one_shape.shape[1] < max_point: # padding 
                tex_pos_one_shape = torch.cat(
                    [tex_pos_one_shape, torch.zeros(
                        1, max_point - tex_pos_one_shape.shape[1], 3,
                        device=tex_pos_one_shape.device, dtype=torch.float32)], dim=1)

            sample_tex_pose_list.append(tex_pos_one_shape)

        tex_pos = torch.cat(sample_tex_pose_list, dim=0) # [batch_size, max_point, 3]
        # tex_coordinates_cano, tex_coordinates_obs, tex_mask, tex_coordinates_mask, tex_max_length = self.canonical_mapping(tex_pos, **canonical_mapping_kwargs)
        tex_mask, tex_coordinates_mask, tex_max_length, tex_coordinates_obs, _ = self.filter_and_pad(tex_pos, self.cano_bbox[None].repeat(tex_pos.shape[0], 1, 1))

        if self.one_3d_generator:
            tex_feat_cano = self.generator.get_texture_prediction(tex_feature, tex_coordinates_obs, ws)
            tex_feat = torch.zeros((batch_size, max_point, tex_feat_cano.shape[-1])).to(self.device)
            tex_feat[tex_mask] = tex_feat_cano[tex_coordinates_mask]
        else:
            raise NotImplementedError

        # if hard_mask is not None:
        final_tex_feat = torch.zeros(
            ws.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], device=tex_feat.device)
        expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
        for i in range(ws.shape[0]):
            final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[i][:n_point_list[i]].reshape(-1)
        # tex_feat = final_tex_feat

        return final_tex_feat.reshape(ws.shape[0], hard_mask.shape[1], hard_mask.shape[2], final_tex_feat.shape[-1])

    def get_texture_prediction_with_eikonal(self, ws, tex_pos, ws_geo, hard_mask, tex_feature, sdf_feature, canonical_mapping_kwargs):
        '''
        Predict Texture given latent codes
        :param ws: Latent code for texture generation
        :param tex_pos: Position we want to query the texture field
        :param ws_geo: latent code for geometry
        :param hard_mask: 2D silhoueete of the rendered image
        :param tex_feature: the triplane feature map
        :return:
        '''
        tex_pos = torch.cat(tex_pos, dim=0)
        
        # if not hard_mask is None:
        tex_pos = tex_pos * hard_mask.float()
        batch_size = tex_pos.shape[0]
        tex_pos = tex_pos.reshape(batch_size, -1, 3)   # [batch_size, res*res, 3]
        # hard_mask [batch_size, res, res, 1]

        ###################
        # We use mask to get the texture location (to save the memory)
        # if hard_mask is not None:
        n_point_list = torch.sum(hard_mask.long().reshape(hard_mask.shape[0], -1), dim=-1) # [batch_size]
        sample_tex_pose_list = []
        max_point = n_point_list.max() # max_length among batch_size number
        expanded_hard_mask = hard_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3) > 0.5 # [batch_size, res*res, 3]
        for i in range(tex_pos.shape[0]):
            tex_pos_one_shape = tex_pos[i][expanded_hard_mask[i]].reshape(1, -1, 3) # [batch_size, n_point_list[i], 3]

            if tex_pos_one_shape.shape[1] < max_point: # padding 
                tex_pos_one_shape = torch.cat(
                    [tex_pos_one_shape, torch.zeros(
                        1, max_point - tex_pos_one_shape.shape[1], 3,
                        device=tex_pos_one_shape.device, dtype=torch.float32)], dim=1)

            sample_tex_pose_list.append(tex_pos_one_shape)

        tex_pos = torch.cat(sample_tex_pose_list, dim=0) # [batch_size, max_point, 3]
        # tex_coordinates_cano, tex_coordinates_obs, tex_mask, tex_coordinates_mask, tex_max_length = self.canonical_mapping(tex_pos, **canonical_mapping_kwargs)
        tex_mask, tex_coordinates_mask, tex_max_length, tex_coordinates_obs, _ = self.filter_and_pad(tex_pos, self.cano_bbox[None].repeat(tex_pos.shape[0], 1, 1))
        
        if self.one_3d_generator:
            tex_feat_cano = self.generator.get_texture_prediction(tex_feature, tex_coordinates_obs, ws)
            tex_feat = torch.zeros((batch_size, max_point, tex_feat_cano.shape[-1])).to(self.device)
            tex_feat[tex_mask] = tex_feat_cano[tex_coordinates_mask]

            sdf_cano, _ = self.generator.get_sdf_def_prediction(
                sdf_feature, 
                ws_geo=ws_geo,
                position=tex_coordinates_obs,
                gradient_detach=False)
            tex_coordinates_obs.requires_grad_(True)

            surface_normal_obs_length = torch.zeros((batch_size, max_point, 1)).to(self.device)

            if tex_coordinates_obs.shape[1] > 0:
                try:
                    surface_normal_cano = get_eikonal_term(tex_coordinates_obs, sdf_cano)
                    surface_normal_cano_length = torch.nan_to_num(torch.linalg.norm(surface_normal_cano, dim=-1, keepdim=True), 0.0)
                    # surface_normal_cano = surface_normal_cano / (surface_normal_cano_length + 1e-5)
                    surface_normal_obs_length[tex_mask] = surface_normal_cano_length[tex_coordinates_mask]
                except:
                    pass
        else:
            raise NotImplementedError

        # if hard_mask is not None:
        final_tex_feat = torch.zeros(
            ws.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], device=tex_feat.device)
        expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5

        surface_normal_length_list = []
        for i in range(ws.shape[0]):
            final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[i][:n_point_list[i]].reshape(-1)
            surface_normal_length_list.append(surface_normal_obs_length[i][:n_point_list[i]].reshape(-1))
        # tex_feat = final_tex_feat

        eikonal_term = torch.cat(surface_normal_length_list)
        eikonal_loss = torch.clamp((eikonal_term - 1.0)**2, 0, 1e6).mean()

        return final_tex_feat.reshape(ws.shape[0], hard_mask.shape[1], hard_mask.shape[2], final_tex_feat.shape[-1]), eikonal_loss

    def get_normal_prediction(self, surface_pos, ws_geo, hard_mask, sdf_feature, canonical_mapping_kwargs, return_eikonal=False):
        '''
        Predict Texture given latent codes
        :param surface_pos: Position we want to query the texture field
        :param ws_geo: latent code for geometry
        :param hard_mask: 2D silhoueete of the rendered image
        :param tex_feature: the triplane feature map
        :return:
        '''
        surface_pos = torch.cat(surface_pos, dim=0)
        
        # if not hard_mask is None:
        surface_pos = surface_pos * hard_mask.float()
        batch_size = surface_pos.shape[0]
        surface_pos = surface_pos.reshape(batch_size, -1, 3)   # [batch_size, res*res, 3]

        ###################
        # We use mask to get the texture location (to save the memory)
        # if hard_mask is not None:
        n_point_list = torch.sum(hard_mask.long().reshape(hard_mask.shape[0], -1), dim=-1) # [batch_size]
        sample_surface_pose_list = []
        max_point = n_point_list.max() # max_length among batch_size number
        expanded_hard_mask = hard_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3) > 0.5 # [batch_size, res*res, 3]
        for i in range(surface_pos.shape[0]):
            surface_pos_one_shape = surface_pos[i][expanded_hard_mask[i]].reshape(1, -1, 3) # [batch_size, n_point_list[i], 3]

            if surface_pos_one_shape.shape[1] < max_point: # padding 
                surface_pos_one_shape = torch.cat(
                    [surface_pos_one_shape, torch.zeros(
                        1, max_point - surface_pos_one_shape.shape[1], 3,
                        device=surface_pos_one_shape.device, dtype=torch.float32)], dim=1)

            sample_surface_pose_list.append(surface_pos_one_shape)

        surface_pos = torch.cat(sample_surface_pose_list, dim=0) # [batch_size, max_point, 3]
        # surface_coordinates_cano, surface_coordinates_rotate_matrix, surface_mask, surface_coordinates_mask, surface_max_length = self.canonical_mapping_normal(surface_pos, **canonical_mapping_kwargs)
        surface_mask, surface_coordinates_mask, surface_max_length, surface_coordinates_obs, _ = self.filter_and_pad(surface_pos, self.cano_bbox[None].repeat(surface_pos.shape[0], 1, 1))

        # surface_coordinates_cano[:, :, 1] -= self.obs_bbox_y_center

        if self.one_3d_generator:
            #normal_feat_cano = self.generator.get_normal_prediction(sdf_feature, surface_coordinates_cano, ws_geo)
            # sdf_cano = self.generator.get_normal_prediction(sdf_feature, surface_coordinates_cano, ws_geo)
            sdf_cano, _ = self.generator.get_sdf_def_prediction(
                sdf_feature, ws_geo=ws_geo,
                position=surface_coordinates_obs,
                gradient_detach=False)
            surface_coordinates_obs.requires_grad_(True)

            surface_normal_obs = torch.zeros((batch_size, max_point, 3)).to(self.device)
            if return_eikonal:
                surface_normal_obs_length = torch.zeros((batch_size, max_point, 1)).to(self.device)

            if surface_coordinates_obs.shape[1] > 0:
                try:
                    surface_normal_cano = get_eikonal_term(surface_coordinates_obs, sdf_cano)
                    surface_normal_cano_length = torch.nan_to_num(torch.linalg.norm(surface_normal_cano, dim=-1, keepdim=True), 0.0)
                    surface_normal_cano =  surface_normal_cano / (surface_normal_cano_length + 1e-5)
                    # NOTE: check normal_feat_transform shape here
                    surface_normal_transform = surface_normal_cano
                    surface_normal_obs[surface_mask] = surface_normal_transform[surface_coordinates_mask]
                    if return_eikonal:
                        surface_normal_obs_length[surface_mask] = surface_normal_cano_length[surface_coordinates_mask]
                except:
                    pass
        else:
            raise NotImplementedError

        # if hard_mask is not None:
        final_surface_normal_obs = torch.zeros(
            batch_size, hard_mask.shape[1] * hard_mask.shape[2], 3, device=self.device)
        expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, 3) > 0.5

        if return_eikonal:
            surface_normal_length_list = []
            
        for i in range(ws_geo.shape[0]):
            final_surface_normal_obs[i][expanded_hard_mask[i]] = surface_normal_obs[i][:n_point_list[i]].reshape(-1)
            if return_eikonal:
                surface_normal_length_list.append(surface_normal_obs_length[i][:n_point_list[i]].reshape(-1))
        # surface_normal_obs = final_surface_normal_obs

        if return_eikonal:
            eikonal_term = torch.cat(surface_normal_length_list)
            eikonal_loss = torch.clamp((eikonal_term - 1.0)**2, 0, 1e6).mean()
        else:
            eikonal_loss = None

        return final_surface_normal_obs.reshape(ws_geo.shape[0], hard_mask.shape[1], hard_mask.shape[2], 3), eikonal_loss


    def render_mesh(self, mesh_v, mesh_f, cam_mv, rendering_resolution=None):
        '''
        Function to render a generated mesh with nvdiffrast
        :param mesh_v: List of vertices for the mesh
        :param mesh_f: List of faces for the mesh
        :param cam_mv:  4x4 rotation matrix
        :return:
        '''
        # TODO: progressive training
        return_value_list = []
        if rendering_resolution is None:
            rendering_resolution = self.img_resolution
        for i_mesh in range(len(mesh_v)):
            return_value = self.dmtet_geometry.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=rendering_resolution,
                hierarchical_mask=False
            )
            return_value_list.append(return_value)

        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask_list, hard_mask_list = torch.cat(return_value['mask'], dim=0), torch.cat(return_value['hard_mask'], dim=0)
        return mask_list, hard_mask_list, return_value

    def extract_3d_shape(
            self, ws, ws_geo=None, texture_resolution=2048,
            **block_kwargs):
        '''
        Extract the 3D shape with texture map with GET3D generator
        :param ws: latent code to control texture generation
        :param ws_geo: latent code to control geometry generation
        :param texture_resolution: the resolution for texure map
        :param block_kwargs:
        :return:
        '''

        # Step 1: predict geometry first
        if self.one_3d_generator:
            sdf_feature, tex_feature = self.generator.get_feature(
                ws[:, :self.generator.tri_plane_synthesis.num_ws_tex],
                ws_geo[:, :self.generator.tri_plane_synthesis.num_ws_geo])
            ws = ws[:, self.generator.tri_plane_synthesis.num_ws_tex:]
            ws_geo = ws_geo[:, self.generator.tri_plane_synthesis.num_ws_geo:]
            mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo, sdf_feature)
        else:
            # mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo)
            raise NotImplementedError

        # Step 2: use x-atlas to get uv mapping for the mesh
        from training.extract_texture_map import xatlas_uvmap
        all_uvs = []
        all_mesh_tex_idx = []
        all_gb_pose = []
        all_uv_mask = []
        if self.dmtet_geometry.renderer.ctx is None:
            # self.dmtet_geometry.renderer.ctx = dr.RasterizeGLContext(device=self.device)
            self.dmtet_geometry.renderer.ctx = dr.RasterizeCudaContext(device=self.device)
        for v, f in zip(mesh_v, mesh_f):
            uvs, mesh_tex_idx, gb_pos, mask = xatlas_uvmap(
                self.dmtet_geometry.renderer.ctx, v, f, resolution=texture_resolution)
            all_uvs.append(uvs)
            all_mesh_tex_idx.append(mesh_tex_idx)
            all_gb_pose.append(gb_pos)
            all_uv_mask.append(mask)

        tex_hard_mask = torch.cat(all_uv_mask, dim=0).float()

        # Step 3: Query the texture field to get the RGB color for texture map
        # we use run one per iteration to avoid OOM error
        all_network_output = []
        for _ws, _all_gb_pose, _ws_geo, _tex_hard_mask in zip(ws, all_gb_pose, ws_geo, tex_hard_mask):
            if self.one_3d_generator:
                tex_feat = self.get_texture_prediction(
                    _ws.unsqueeze(dim=0), [_all_gb_pose],
                    _ws_geo.unsqueeze(dim=0).detach(),
                    _tex_hard_mask.unsqueeze(dim=0),
                    tex_feature)
            else:
                raise NotImplementedError
                # tex_feat = self.get_texture_prediction(
                #     _ws.unsqueeze(dim=0), [_all_gb_pose],
                #     _ws_geo.unsqueeze(dim=0).detach(),
                #     _tex_hard_mask.unsqueeze(dim=0))
            background_feature = torch.zeros_like(tex_feat)
            # Merge them together
            img_feat = tex_feat * _tex_hard_mask.unsqueeze(dim=0) + background_feature * (
                    1 - _tex_hard_mask.unsqueeze(dim=0))
            network_out = self.to_rgb(img_feat.permute(0, 3, 1, 2), _ws.unsqueeze(dim=0)[:, -1])
            all_network_output.append(network_out)
        network_out = torch.cat(all_network_output, dim=0)
        return mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out

    def get_canonical_mapping_info(self, c):
        """
        label info: 
            0:16    world2camera_matrix shape (16,)
            16:19   global_orient       shape (3,)
            19:88   body_pose           shape (69,)
            88:98   betas               shape (10,)
        """

        c_global_orient = c[:, 16:19]
        c_body_pose=  c[:, 19:88]
        c_betas = c[:, 88:98]

        with torch.no_grad():
             # Observation Space
            body_model_params = {'global_orient':c_global_orient, 'body_pose': c_body_pose, 'betas': c_betas}
            body_model_out = self.body_model(**body_model_params, return_verts=True)
            verts = body_model_out['vertices']
            triangles_obs = face_vertices(verts, self.faces, c.device)

            verts_transform = body_model_out['vertices_transform']
            shape_offsets = body_model_out['shape_offsets']
            pose_offsets = body_model_out['pose_offsets']
            del body_model_out

            verts_transform_canonical = self.canonical_smpl_info['verts_transform_canonical'].repeat(c.shape[0], 1, 1, 1)
            shape_offsets_canonical = self.canonical_smpl_info['shape_offsets_canonical'].repeat(c.shape[0], 1, 1)
            pose_offsets_canonical = self.canonical_smpl_info['pose_offsets_canonical'].repeat(c.shape[0], 1, 1)

            # Compute transformation from observation to canonical space
            ober2cano_transform = self.calc_ober2cano_transform(
                verts_transform, verts_transform_canonical,
                shape_offsets, shape_offsets_canonical,
                pose_offsets, pose_offsets_canonical
            )

            # Compute body bbox
            body_min = torch.amin(verts, dim=1)
            body_max = torch.amax(verts, dim=1)
            # extend box by a pre-defined ratio
            ratio = 0.2  # TODO: set as an arugment
            body_bbox = torch.stack([(1+ratio)*body_min-ratio*body_max, (1+ratio)*body_max-ratio*body_min], dim=1)  # Bx2x3

            canonical_mapping_kwargs = {
                'ober2cano_transform': ober2cano_transform,
                'verts': verts,
                'triangles_obs': triangles_obs,
                'body_bbox': body_bbox,
            }
            return canonical_mapping_kwargs

    def generate_normal_map(
            self, ws_tex, c, ws_geo, return_eikonal=False, rendering_resolution=None, **block_kwargs):
        '''
        Main function of our Generator. Given two latent code `ws_tex` for texture generation
        `ws_geo` for geometry generation. It first generate 3D mesh, then render it into 2D image
        with given `camera` or sampled from a prior distribution of camera.
        :param ws_tex: latent code for texture
        :param camera: camera to render generated 3D shape
        :param ws_geo: latent code for geometry
        :param block_kwargs:
        :return:
        '''
        run_n_view = self.n_views
        cam_mv = c[:, :self.c_cam_dim].view(ws_tex.shape[0], run_n_view, 4, 4)
        # Generate 3D mesh first
        if self.one_3d_generator:
            sdf_feature, _ = self.generator.get_feature(
                ws_tex[:, :self.generator.tri_plane_synthesis.num_ws_tex],
                ws_geo[:, :self.generator.tri_plane_synthesis.num_ws_geo])
            # ws_tex_sr = ws_tex[:, -1:]
            # ws_tex = ws_tex[:, self.generator.tri_plane_synthesis.num_ws_tex:]
            ws_geo = ws_geo[:, self.generator.tri_plane_synthesis.num_ws_geo:]
            canonical_mapping_kwargs = self.get_canonical_mapping_info(c)
            mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo, sdf_feature, canonical_mapping_kwargs)
        else:
            raise NotImplementedError
            #mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo)
        # Render the mesh into 2D image (get 3d position of each image plane)
        # cam_mv: (batch_size, n_views, 4, 4)
        antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv, rendering_resolution=rendering_resolution)

        mask_pyramid = None

        surface_pos = return_value['tex_pos']
        
        surface_hard_mask = hard_mask
        surface_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in surface_pos]
        # surface_hard_mask = torch.cat(
        #     [torch.cat(
        #         [surface_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
        #          for i_view in range(run_n_view)], dim=2)
        #         for i in range(ws_geo.shape[0])], dim=0)
        
        # Querying the geometry field to predict the normal feature for each pixel on the image
        if self.one_3d_generator:
            normal_obs_feat, eikonal_loss = self.get_normal_prediction(
                surface_pos, ws_geo, surface_hard_mask,
                sdf_feature, canonical_mapping_kwargs=canonical_mapping_kwargs,
                return_eikonal=return_eikonal)
        else:
            raise NotImplementedError

        background_feature = torch.zeros_like(normal_obs_feat)
        # Merge them together
        normal_feat = normal_obs_feat * surface_hard_mask + background_feature * (1 - surface_hard_mask)
        normal_feat = normal_feat.permute(0, 3, 1, 2)
        
        if normal_feat.shape[-1] < self.img_resolution:
            normal_feat = torch.nn.functional.interpolate(
                normal_feat, size=(self.img_resolution, self.img_resolution),
                mode='bilinear', align_corners=False)
            
            antilias_mask = antilias_mask.permute(0, 3, 1, 2)
            antilias_mask = torch.nn.functional.interpolate(
                antilias_mask, size=(self.img_resolution, self.img_resolution),
                mode='bilinear', align_corners=False)
            antilias_mask = antilias_mask.permute(0, 2, 3, 1)

        # We should split it back to the original image shape
        # normal_feat = torch.cat(
        #     [torch.cat(
        #         [normal_feat[i:i + 1, :, self.img_resolution * i_view: self.img_resolution * (i_view + 1)]
        #          for i_view in range(run_n_view)], dim=0) for i in range(len(return_value['tex_pos']))], dim=0)

        normal_img = normal_feat
        img_buffers_viz = None

        final_img = torch.cat([normal_img, antilias_mask.permute(0, 3, 1, 2)], dim=1)

        # return img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_buffers_viz, \
        #        mask_pyramid, tex_hard_mask, sdf_reg_loss, return_value
        return final_img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_buffers_viz, \
               mask_pyramid, surface_hard_mask, sdf_reg_loss, eikonal_loss, return_value

    def generate(
            self, ws_tex, c, ws_geo, return_eikonal=False, rendering_resolution=None, **block_kwargs):
        '''
        Main function of our Generator. Given two latent code `ws_tex` for texture generation
        `ws_geo` for geometry generation. It first generate 3D mesh, then render it into 2D image
        with given `camera` or sampled from a prior distribution of camera.
        :param ws_tex: latent code for texture
        :param camera: camera to render generated 3D shape
        :param ws_geo: latent code for geometry
        :param block_kwargs:
        :return:
        '''
        run_n_view = self.n_views
        cam_mv = c[:, :self.c_cam_dim].view(ws_tex.shape[0], run_n_view, 4, 4)
        # Generate 3D mesh first
        if self.one_3d_generator:
            sdf_feature, tex_feature = self.generator.get_feature(
                ws_tex[:, :self.generator.tri_plane_synthesis.num_ws_tex],
                ws_geo[:, :self.generator.tri_plane_synthesis.num_ws_geo])
            ws_tex_sr = ws_tex[:, -1:]
            ws_tex = ws_tex[:, self.generator.tri_plane_synthesis.num_ws_tex:]
            ws_geo = ws_geo[:, self.generator.tri_plane_synthesis.num_ws_geo:]
            canonical_mapping_kwargs = self.get_canonical_mapping_info(c)
            mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo, sdf_feature, canonical_mapping_kwargs)
        else:
            raise NotImplementedError
            
        # Render the mesh into 2D image (get 3d position of each image plane)
        # cam_mv: (batch_size, n_views, 4, 4)
        antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv, rendering_resolution=rendering_resolution)
        mask_pyramid = None
        tex_pos = return_value['tex_pos']
        
        tex_hard_mask = hard_mask
        tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
        # tex_hard_mask = torch.cat(
        #     [torch.cat(
        #         [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
        #          for i_view in range(run_n_view)], dim=2)
        #         for i in range(ws_tex.shape[0])], dim=0)
        
        # Querying the texture field to predict the texture feature for each pixel on the image
        if self.one_3d_generator:
            if return_eikonal:
                # get_texture_prediction_with_eikonal(ws, tex_pos, ws_geo, hard_mask, tex_feature, sdf_feature, canonical_mapping_kwargs)
                tex_feat, eikonal_loss = self.get_texture_prediction_with_eikonal(
                    ws_tex, tex_pos, ws_geo, tex_hard_mask,
                    tex_feature, sdf_feature, canonical_mapping_kwargs=canonical_mapping_kwargs)
            else:
                eikonal_loss = None
                # get_texture_prediction(ws, tex_pos, hard_mask, tex_feature, canonical_mapping_kwargs)
                tex_feat = self.get_texture_prediction(
                    ws_tex, tex_pos, tex_hard_mask,
                    tex_feature, canonical_mapping_kwargs=canonical_mapping_kwargs)
        else:
            raise NotImplementedError
        background_feature = torch.zeros_like(tex_feat)

        # Merge them together
        img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)
        img_feat = img_feat.permute(0, 3, 1, 2)

        if img_feat.shape[-1] < self.img_resolution:
            img_feat = torch.nn.functional.interpolate(
                img_feat, size=(self.img_resolution, self.img_resolution),
                mode='bilinear', align_corners=False)
            
            antilias_mask = antilias_mask.permute(0, 3, 1, 2)
            antilias_mask = torch.nn.functional.interpolate(
                antilias_mask, size=(self.img_resolution, self.img_resolution),
                mode='bilinear', align_corners=False)
            antilias_mask = antilias_mask.permute(0, 2, 3, 1)

        # We should split it back to the original image shape
        # img_feat = torch.cat(
        #     [torch.cat(
        #         [img_feat[i:i + 1, :, self.img_resolution * i_view: self.img_resolution * (i_view + 1)]
        #          for i_view in range(run_n_view)], dim=0) for i in range(len(return_value['tex_pos']))], dim=0)

        # ws_list = [ws_tex[i].unsqueeze(dim=0).expand(return_value['tex_pos'][i].shape[0], -1, -1) for i in
        #            range(len(return_value['tex_pos']))]
        # ws = torch.cat(ws_list, dim=0).contiguous()

        # Predict the RGB color for each pixel (self.to_rgb is 1x1 convolution)
        if self.feat_channel > 3:
            # network_out = self.to_rgb(img_feat.permute(0, 3, 1, 2), ws[:, -1])
            # network_out = self.rgb_decoder(img_feat)
            network_out = img_feat
        else:
            network_out = img_feat.permute(0, 3, 1, 2)

        # rgb_feat =  network_out.permute(0, 3, 1, 2)
        rgb_feat = network_out
        img_buffers_viz = None

        if self.render_type == 'neural_render':
            # img = img[:, :3]
            # img = rgb_feat[:, :3]
            img = self.to_rgb(rgb_feat, ws_tex_sr[:, -1])
        else:
            raise NotImplementedError
        
        sr_image = self.superresolution(
            img, rgb_feat, ws_tex_sr, noise_mode='none')

        final_img = torch.cat([sr_image, antilias_mask.permute(0, 3, 1, 2)], dim=1)

        return final_img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_buffers_viz, \
               mask_pyramid, tex_hard_mask, sdf_reg_loss, eikonal_loss, return_value

    def forward_normal(self, ws, c, ws_geo, return_shape=False, return_eikonal=False, rendering_resolution=None, **block_kwargs):
        normal_img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, mask_pyramid, \
        tex_hard_mask, sdf_reg_loss, eikonal_loss, render_return_value = self.generate_normal_map(ws, c, ws_geo, return_eikonal=return_eikonal, rendering_resolution=rendering_resolution, **block_kwargs)
        if return_shape:
            return normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value
        return normal_img, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value

    def forward(self, ws, c, ws_geo, return_shape=False, return_eikonal=False, rendering_resolution=None, **block_kwargs):
        img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, mask_pyramid, \
        tex_hard_mask, sdf_reg_loss, eikonal_loss, render_return_value = self.generate(ws, c, ws_geo, return_eikonal=return_eikonal, rendering_resolution=rendering_resolution, **block_kwargs)
        if return_shape:
            return img, sdf, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value
        return img, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value


@persistence.persistent_class
class GeneratorDMTETMesh(torch.nn.Module):
    def __init__(
            self,
            z_dim,  # Input latent (Z) dimensionality.
            c_dim,  # Conditioning label (C) dimensionality.
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output resolution.
            img_channels,  # Number of output color channels.
            mapping_kwargs={},  # Arguments for MappingNetwork.
            use_style_mixing=False,  # Whether use stylemixing or not
            **synthesis_kwargs,  # Arguments for SynthesisNetpwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.c_cam_dim = 16
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.device = synthesis_kwargs['device']
        self.use_style_mixing = use_style_mixing

        self.synthesis = DMTETSynthesisNetwork(
            w_dim=w_dim, img_resolution=img_resolution, img_channels=self.img_channels,
            **synthesis_kwargs)

        if self.synthesis.one_3d_generator:
            self.num_ws = self.synthesis.generator.num_ws_tex
            self.num_ws_geo = self.synthesis.generator.num_ws_geo
        else:
            raise NotImplementedError

        self.mapping = MappingNetwork(
            z_dim=z_dim, c_dim=0, w_dim=w_dim, num_ws=self.num_ws,
            device=self.synthesis.device, **mapping_kwargs)
        self.mapping_geo = MappingNetwork(
            z_dim=z_dim, c_dim=0, w_dim=w_dim, num_ws=self.num_ws_geo,
            device=self.synthesis.device, **mapping_kwargs)

    def update_w_avg(self, c=None):
        # Update the the average latent to compute truncation
        self.mapping.update_w_avg(self.device, c)
        self.mapping_geo.update_w_avg(self.device, c)

    def generate_3d_mesh(
            self, geo_z, tex_z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False,
            with_texture=True, use_style_mixing=False, use_mapping=True, **synthesis_kwargs):
        '''
        This function generates a 3D mesh with given geometry latent code (geo_z) and texture
        latent code (tex_z), it can also generate a texture map is setting `with_texture` to be True.
        :param geo_z: lantent code for geometry
        :param tex_z: latent code for texture
        :param c: None is default
        :param truncation_psi: the trucation for the latent code
        :param truncation_cutoff: Where to cut the truncation
        :param update_emas: False is default
        :param with_texture: Whether generating texture map along with the 3D mesh
        :param use_style_mixing: Whether use style mixing for generation
        :param use_mapping: Whether we need to use mapping network to map the latent code
        :param synthesis_kwargs:
        :return:
        '''
        if not with_texture:
            c_cam_perm = torch.zeros_like(c[:, :self.c_cam_dim])
            # self.style_mixing_prob = 0.9
            # Mapping the z to w space
            if use_mapping:
                ws_geo = self.mapping_geo(
                    geo_z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                    update_emas=update_emas)
            else:
                ws_geo = geo_z

            if self.synthesis.one_3d_generator:
                # For this model, we first generate the feature map for it
                ws_tex = self.mapping(geo_z, c_cam_perm, truncation_psi=truncation_psi)  # we didn't use it during inference
                sdf_feature, tex_feature = self.synthesis.generator.get_feature(
                    ws_tex[:, :self.synthesis.generator.tri_plane_synthesis.num_ws_tex],
                    ws_geo[:, :self.synthesis.generator.tri_plane_synthesis.num_ws_geo])
                ws_tex = ws_tex[:, self.synthesis.generator.tri_plane_synthesis.num_ws_tex:]
                ws_geo = ws_geo[:, self.synthesis.generator.tri_plane_synthesis.num_ws_geo:]
                mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.synthesis.get_geometry_prediction(ws_geo, sdf_feature)
            else:
                raise NotImplementedError
            return mesh_v, mesh_f

        if use_mapping:
            ws = self.mapping(
                tex_z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            ws_geo = self.mapping_geo(
                geo_z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                update_emas=update_emas)
        else:
            ws = tex_z
            ws_geo = geo_z

        all_mesh = self.synthesis.extract_3d_shape(ws, ws_geo, )

        return all_mesh

    def generate_3d(
            self, z, geo_z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, 
            # camera=None,
            generate_no_light=False,
            **synthesis_kwargs):
        c_cam_perm = torch.zeros_like(c[:, :self.c_cam_dim])

        ws = self.mapping(
            z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        ws_geo = self.mapping_geo(
            geo_z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
            update_emas=update_emas)

        img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, mask_pyramid, tex_hard_mask, \
        sdf_reg_loss, eikonal_loss, render_return_value = self.synthesis.generate(
            ws, c=c,
            ws_geo=ws_geo,
            **synthesis_kwargs)
        
        ##############################
        # final_img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_buffers_viz, \
        #        mask_pyramid, surface_hard_mask, sdf_reg_loss, eikonal_loss, return_value
        normal_img, _, _, _, _, _, _, _, _, _, _, _, _ = self.synthesis.generate_normal_map(
            ws, c, 
            ws_geo=ws_geo,
            **synthesis_kwargs)
        ##############################
        if generate_no_light:
            # return img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask
            return img, mask, normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, tex_hard_mask
        # return img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, tex_hard_mask
        return img, mask, normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, tex_hard_mask

    def forward(
            self, z=None, c=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, use_style_mixing=False,
            geo_z=None,
            **synthesis_kwargs):
        '''
        The function generate rendered 2D image of 3D shapes using the given sampled z for texture and geometry
        :param z:  sample z for textur generation
        :param c: None is default
        :param truncation_psi: truncation value
        :param truncation_cutoff: where to cut the truncation
        :param update_emas: False is default
        :param use_style_mixing: whether use style-mixing
        :param geo_z: sample z for geometry generation
        :param synthesis_kwargs:
        :return:
        '''
        c_cam_perm = torch.zeros_like(c[:, :self.c_cam_dim])

        ws = self.mapping(
            z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

        if geo_z is None:
            geo_z = torch.randn_like(z)

        ws_geo = self.mapping_geo(
            geo_z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
            update_emas=update_emas)

        img, sdf, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, _, _, _ = self.synthesis(
            ws=ws, 
            c=c,
            ws_geo=ws_geo,
            update_emas=update_emas,
            return_shape=True,
        )
        return img
