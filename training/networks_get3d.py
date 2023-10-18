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
from torchvision.ops import roi_align
import skimage.measure
from torch_utils import persistence
import nvdiffrast.torch as dr
from uni_rep.rep_3d.dmtet import DMTetGeometry
from uni_rep.camera.perspective_camera import PerspectiveCamera
from uni_rep.render.neural_render import NeuralRender, xfm_points
from training.discriminator_architecture import Discriminator, MultiDiscriminator
from training.geometry_predictor import  MappingNetwork, ToRGBLayer, \
    TriPlaneTexGeo, SuperresolutionHybrid4X, SuperresolutionHybrid8X, SuperresolutionHybrid16X

from pytorch3d.ops.knn import knn_points

from smplx import create
from training.smpl_utils import get_canonical_pose, face_vertices, cal_sdf_batch, \
    batch_index_select, batch_transform, batch_transform_normal, get_eikonal_term, create_samples


@persistence.persistent_class
class DMTETSynthesisNetwork(torch.nn.Module):
    def __init__(
            self,
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output image resolution.
            img_channels,  # Number of color channels.
            unit_2norm,
            device='cuda',
            use_normal_offset=False,
            tet_res=64,  # Resolution for tetrahedron grid
            render_type='neural_render',  # neural type
            camera_type='blender',
            n_views=1,
            tri_plane_resolution=128,
            deformation_multiplier=2.0,
            feat_channel=128,
            mlp_latent_channel=256,
            dmtet_scale=2.0, # adjust the scale according to the canonical space or observation space?
            inference_noise_mode='random',
            one_3d_generator=False,
            with_sr=True,
            part_disc=True,
            **block_kwargs,  # Arguments for SynthesisBlock.
    ):  #
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.device = device
        self.one_3d_generator = one_3d_generator
        self.inference_noise_mode = inference_noise_mode
        self.dmtet_scale = dmtet_scale
        self.deformation_multiplier = deformation_multiplier

        self.n_freq_posenc_geo = 1
        self.render_type = render_type
        
        self.obs_bbox_y_max = 0.96
        self.obs_bbox_y_min = -1.33
        self.dmtet_scale = self.obs_bbox_y_max - self.obs_bbox_y_min
        self.obs_bbox_y_center = 0.5 * (self.obs_bbox_y_max + self.obs_bbox_y_min)
        self.obs_bbox_center_coordinates = torch.tensor([0.0, self.obs_bbox_y_center, 0.0]).reshape(1, 1, 3).float().to(self.device)

        self.cano_bbox_length = 2.0 # range: [-1.185, 0.815]
        self.w_dim = w_dim
        self.c_cam_dim = 16
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.n_views = n_views
        self.grid_res = tet_res

        # Camera defination, we follow the defination from Blender (check the render_shapenet_data/rener_shapenet.py for more details)
        # fovy = np.arctan(32 / 2 / 35) * 2
        # fovyangle = fovy / np.pi * 180.0
        # dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=self.device)

        blender_fovy = np.arctan(32 / 2 / 35) * 2
        blender_fovyangle = blender_fovy / np.pi * 180.0
        blender_camera = PerspectiveCamera(fovy=blender_fovyangle, device=self.device)

        smpl_fovy = 0.20408864225852996 # radians
        smpl_fovyangle = smpl_fovy / np.pi * 180.0
        smpl_camera = PerspectiveCamera(fovy=smpl_fovyangle, device=self.device)

        self.camera_type = camera_type

        # Renderer we used.
        dmtet_renderer = NeuralRender(device, blender_camera_model=blender_camera, smpl_camera_model=smpl_camera)

        # Geometry class for DMTet
        self.dmtet_geometry = DMTetGeometry(
            grid_res=self.grid_res, 
            scale=self.dmtet_scale, 
            renderer=dmtet_renderer, 
            render_type=render_type,
            device=self.device)

        self.feat_channel = feat_channel
        self.mlp_latent_channel = mlp_latent_channel
        # self.use_tri_plane = use_tri_plane
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
                tri_plane_resolution=tri_plane_resolution,
                device=self.device,
                mlp_latent_channel=self.mlp_latent_channel,
                **block_kwargs)
        else:
            raise NotImplementedError

        self.channels_last = False
        self.with_sr = with_sr
        self.part_disc = part_disc

        if self.feat_channel > 3:
            # Final layer to convert the texture field to RGB color, this is only a fully connected layer.
            self.to_rgb = ToRGBLayer(
                self.feat_channel, self.img_channels, w_dim=w_dim,
                conv_clamp=256, channels_last=self.channels_last, device=self.device)

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
        elif self.img_resolution == 1024:
            self.superresolution = SuperresolutionHybrid16X(
                channels=self.feat_channel, 
                img_resolution=self.img_resolution, 
                sr_num_fp16_res=4, 
                sr_antialias=True, 
                channel_base=32768, 
                channel_max=512, 
                fused_modconv_default='inference_only')
        else:
            raise NotImplementedError

        # self.glctx = None
        self.unit_2norm = unit_2norm
        self.use_normal_offset = use_normal_offset

         # Define SMPL BODY MODEL
        self.body_model = create(model_path='./smplx/models', model_type='smpl', gender='neutral').to(self.device)
        # Define X-Pose
        pose_canonical = torch.from_numpy(np.array(get_canonical_pose())).float().to(self.device)
        lbs_weights = self.body_model.lbs_weights.to(self.device)
        faces = torch.from_numpy(self.body_model.faces.astype(np.int64)).to(self.device)
        zero_smpl_beta = torch.tensor([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        ]).float().to(self.device)
        
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
                'pose_offsets_canonical': pose_offsets_canonical
                 # 'faces': self.faces,
                # 'lbs_weights': self.lbs_weights,
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
            neighbs_dist, neighbs, _ = knn_points(coords, verts, K=k_neigh)
            neighbs_dist = torch.sqrt(neighbs_dist)

        coords_neighbs_transform_inv = batch_index_select(verts_transform_inv, neighbs, self.device) # (bs, n_rays*K, k_neigh, 4, 4)
        coords_transform_inv = coords_neighbs_transform_inv.squeeze(2)

        return neighbs_dist, coords_transform_inv

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
            init_position = init_position + self.obs_bbox_center_coordinates
            sample_coordinates_cano, sample_coordinates_obs, mask, coordinates_mask, max_length = self.canonical_mapping(init_position, **canonical_mapping_kwargs)

            batch_cano_body_sdf = cal_sdf_batch(
                self.canonical_smpl_info['verts_canonical'].repeat(batch_size, 1, 1), 
                self.faces,
                self.canonical_smpl_info['triangles_canonical'].repeat(batch_size, 1, 1, 1), 
                sample_coordinates_cano, 
                self.device) # [batch_size, 98653, 1]
                    
            sample_coordinates_cano = sample_coordinates_cano - self.obs_bbox_center_coordinates
            res_sdf_cano, deformation_cano = self.generator.get_sdf_def_prediction(
                sdf_feature, ws_geo=ws,
                position=sample_coordinates_cano)
            
            sdf_cano = res_sdf_cano + batch_cano_body_sdf
            
            sdf = torch.zeros((batch_size, num_pts, 1)).to(self.device) + 10.0
            deformation = torch.zeros((batch_size, num_pts, 3)).to(self.device)

            sdf[mask] = sdf_cano[coordinates_mask]
            deformation[mask] = deformation_cano[coordinates_mask]
        else:
            raise NotImplementedError

        # Step 2: Normalize the deformation to avoid the flipped triangles.
        deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation)

        sdf_reg_loss = torch.zeros(sdf.shape[0], device=sdf.device, dtype=torch.float32)

        ####
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

    def extract_mesh_with_marching_cubes(
            self, ws_tex, c, ws_geo=None, 
            with_texture=False, shape_res=256,
            texture_resolution=2048, return_depth=False,
             **block_kwargs):

        run_n_view = self.n_views
        cam_mv = c[:, :self.c_cam_dim].view(ws_tex.shape[0], run_n_view, 4, 4)
        batch_size = ws_geo.shape[0]
        # Generate 3D mesh first
        sdf_feature, tex_feature = self.generator.get_feature(
            ws_tex[:, :self.generator.tri_plane_synthesis.num_ws_tex],
            ws_geo[:, :self.generator.tri_plane_synthesis.num_ws_geo])
        ws_tex = ws_tex[:, self.generator.tri_plane_synthesis.num_ws_tex:]
        ws_geo = ws_geo[:, self.generator.tri_plane_synthesis.num_ws_geo:]
        canonical_mapping_kwargs = self.get_canonical_mapping_info(c)
        init_position, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=self.dmtet_scale) 
        init_position = init_position.to(ws_geo.device)
        num_pts = init_position.shape[1]
        # init_position[:, :, 1] += self.obs_bbox_y_center
        init_position = init_position + self.obs_bbox_center_coordinates
        sample_coordinates_cano, sample_coordinates_obs, mask, coordinates_mask, max_length = self.canonical_mapping(init_position, **canonical_mapping_kwargs)
        # if self.body_sdf_from_cano:
        batch_cano_body_sdf = cal_sdf_batch(
            self.canonical_smpl_info['verts_canonical'].repeat(batch_size, 1, 1), 
            self.faces,
            self.canonical_smpl_info['triangles_canonical'].repeat(batch_size, 1, 1, 1), 
            sample_coordinates_cano, 
            self.device) # [batch_size, 98653, 1]
                
        # sample_coordinates_cano[:, :, 1] -= self.obs_bbox_y_center
        sample_coordinates_cano = sample_coordinates_cano - self.obs_bbox_center_coordinates
        res_sdf_cano, deformation_cano = self.generator.get_sdf_def_prediction(
            sdf_feature, ws_geo=ws_geo,
            position=sample_coordinates_cano)
        
        sdf_cano = res_sdf_cano + batch_cano_body_sdf
        
        sdf = torch.zeros((batch_size, num_pts, 1)).to(self.device) + 10.0
        deformation = torch.zeros((batch_size, num_pts, 3)).to(self.device)

        sdf[mask] = sdf_cano[coordinates_mask]
        deformation[mask] = deformation_cano[coordinates_mask]

        sdf_samples = sdf.reshape((shape_res, shape_res, shape_res)).detach().cpu().numpy()
        verts, faces, _, _ = skimage.measure.marching_cubes(sdf_samples,level=0.0)
        verts[:,0] = (verts[:, 0] / float(shape_res) - 0.5) * self.dmtet_scale
        verts[:,1] = (verts[:, 1] / float(shape_res) - 0.5) * self.dmtet_scale + self.obs_bbox_y_center
        verts[:,2] = (verts[:, 2] / float(shape_res) - 0.5) * self.dmtet_scale

        mesh_v = [torch.from_numpy(verts.copy()).to(self.device)]
        mesh_f = [torch.from_numpy(faces.copy()).to(self.device)]

        if return_depth:
            _, _, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv)
            depth = return_value['depth']
            depth = torch.cat(depth).permute(0, 3, 1, 2)

        if not with_texture:
            if return_depth:
                return mesh_v, mesh_f, depth
            else:
                return mesh_v, mesh_f
        else:
            # Step 2: use x-atlas to get uv mapping for the mesh
            from training.extract_texture_map import xatlas_uvmap
            all_uvs = []
            all_mesh_tex_idx = []
            all_gb_pose = []
            all_uv_mask = []
            if self.dmtet_geometry.renderer.ctx is None:
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
            for _ws, _all_gb_pose, _tex_hard_mask in zip(ws_tex, all_gb_pose, tex_hard_mask):
                if self.one_3d_generator:
                    tex_feat = self.get_texture_prediction(
                        _ws.unsqueeze(dim=0), [_all_gb_pose],
                        _tex_hard_mask.unsqueeze(dim=0),
                        tex_feature, 
                        canonical_mapping_kwargs=canonical_mapping_kwargs)
                else:
                    raise NotImplementedError

                background_feature = torch.zeros_like(tex_feat)
                # Merge them together
                img_feat = tex_feat * _tex_hard_mask.unsqueeze(dim=0) + background_feature * (
                        1 - _tex_hard_mask.unsqueeze(dim=0))
                network_out = img_feat
                rgb_feat =  network_out.permute(0, 3, 1, 2)

                if self.render_type == 'neural_render':
                    img = self.to_rgb(rgb_feat, _ws.unsqueeze(dim=0)[:, -1])
                else:
                    raise NotImplementedError

                if self.with_sr:
                    sr_image = self.superresolution(
                        img, rgb_feat, _ws.unsqueeze(dim=0), noise_mode='none')
                else:
                    sr_image = img

                all_network_output.append(sr_image)
            network_out = torch.cat(all_network_output, dim=0)
            if return_depth:
                return mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out, depth
            else:
                return mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out

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
            verts[:, 1] += self.obs_bbox_y_center
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
        tex_coordinates_cano, tex_coordinates_obs, tex_mask, tex_coordinates_mask, tex_max_length = self.canonical_mapping(tex_pos, **canonical_mapping_kwargs)

        tex_coordinates_cano = tex_coordinates_cano - self.obs_bbox_center_coordinates

        if self.one_3d_generator:
            tex_feat_cano = self.generator.get_texture_prediction(tex_feature, tex_coordinates_cano, ws)
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
        tex_coordinates_cano, tex_coordinates_obs, tex_mask, tex_coordinates_mask, tex_max_length = self.canonical_mapping(tex_pos, **canonical_mapping_kwargs)

        tex_coordinates_cano = tex_coordinates_cano - self.obs_bbox_center_coordinates

        if self.one_3d_generator:
            tex_feat_cano = self.generator.get_texture_prediction(tex_feature, tex_coordinates_cano, ws)
            tex_feat = torch.zeros((batch_size, max_point, tex_feat_cano.shape[-1])).to(self.device)
            tex_feat[tex_mask] = tex_feat_cano[tex_coordinates_mask]

            sdf_cano, _ = self.generator.get_sdf_def_prediction(
                sdf_feature, 
                ws_geo=ws_geo,
                position=tex_coordinates_cano,
                gradient_detach=False)
            tex_coordinates_cano.requires_grad_(True)

            surface_normal_obs_length = torch.zeros((batch_size, max_point, 1)).to(self.device)

            if tex_coordinates_cano.shape[1] > 0:
                try:
                    surface_normal_cano = get_eikonal_term(tex_coordinates_cano, sdf_cano)
                    surface_normal_cano_length = torch.nan_to_num(torch.linalg.norm(surface_normal_cano, dim=-1, keepdim=True), 0.0)
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
        surface_coordinates_cano, surface_coordinates_rotate_matrix, surface_mask, surface_coordinates_mask, surface_max_length = self.canonical_mapping_normal(surface_pos, **canonical_mapping_kwargs)

        if self.one_3d_generator:
            surface_normal_obs = torch.zeros((batch_size, max_point, 3)).to(self.device)
            if return_eikonal:
                surface_normal_obs_length = torch.zeros((batch_size, max_point, 1)).to(self.device)

            if surface_coordinates_cano.shape[1] > 0:
                try:
                    surface_coordinates_cano.requires_grad_(True)
                    batch_cano_body_sdf = cal_sdf_batch(
                        self.canonical_smpl_info['verts_canonical'].repeat(batch_size, 1, 1), 
                        self.faces,
                        self.canonical_smpl_info['triangles_canonical'].repeat(batch_size, 1, 1, 1), 
                        surface_coordinates_cano, 
                        self.device) # [batch_size, 98653, 1]
                    surface_coordinates_cano = surface_coordinates_cano - self.obs_bbox_center_coordinates

                    res_sdf_cano, _ = self.generator.get_sdf_def_prediction(
                        sdf_feature, ws_geo=ws_geo,
                        position=surface_coordinates_cano,
                        gradient_detach=False)

                    sdf_cano = res_sdf_cano + batch_cano_body_sdf
                    surface_normal_cano = get_eikonal_term(surface_coordinates_cano, sdf_cano)
                    surface_normal_cano = torch.nan_to_num(surface_normal_cano, nan=0.0)
                    surface_normal_cano_length = torch.linalg.norm(surface_normal_cano, dim=-1, keepdim=True)

                    if self.use_normal_offset:
                        surface_normal_cano_offset = torch.tanh(self.generator.get_normal_prediction(sdf_feature, surface_coordinates_cano, ws_geo))
                        surface_normal_cano = surface_normal_cano + surface_normal_cano_offset

                    if self.unit_2norm:
                        surface_normal_cano = surface_normal_cano / (surface_normal_cano_length + 1e-5)
                        surface_normal_cano = torch.clamp(surface_normal_cano, -1.0, 1.0)
                    # NOTE: check normal_feat_transform shape here
                    surface_normal_transform = torch.matmul(surface_coordinates_rotate_matrix, surface_normal_cano.unsqueeze(-1)).squeeze(-1)
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


        if return_eikonal:
            eikonal_term = torch.cat(surface_normal_length_list)
            eikonal_loss = torch.clamp((eikonal_term - 1.0)**2, 0, 1e6).mean()
        else:
            eikonal_loss = None

        return final_surface_normal_obs.reshape(ws_geo.shape[0], hard_mask.shape[1], hard_mask.shape[2], 3), eikonal_loss


    def render_mesh(self, mesh_v, mesh_f, cam_mv):
        '''
        Function to render a generated mesh with nvdiffrast
        :param mesh_v: List of vertices for the mesh
        :param mesh_f: List of faces for the mesh
        :param cam_mv:  4x4 rotation matrix
        :return:
        '''
        return_value_list = []
        for i_mesh in range(len(mesh_v)):
            return_value = self.dmtet_geometry.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=self.img_resolution,
                hierarchical_mask=False,
                camera_type=self.camera_type
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
            raise NotImplementedError

        # Step 2: use x-atlas to get uv mapping for the mesh
        from training.extract_texture_map import xatlas_uvmap
        all_uvs = []
        all_mesh_tex_idx = []
        all_gb_pose = []
        all_uv_mask = []
        if self.dmtet_geometry.renderer.ctx is None:
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
            88:98  betas                shape (10,)
        """

        c_global_orient = c[:, 16:19]
        c_body_pose=  c[:, 19:88]
        c_betas = c[:, 88:98]

        with torch.no_grad():
             # Observation Space
            body_model_params = {'global_orient':c_global_orient, 'body_pose': c_body_pose, 'betas': c_betas}
            body_model_out = self.body_model(**body_model_params, return_verts=True)
            verts = body_model_out['vertices']

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
                'body_bbox': body_bbox,
            }
            return canonical_mapping_kwargs

    def get_canonical_mapping_quick(self, c):
        """
        label info: 
            0:16    world2camera_matrix shape (16,)
            16:19   global_orient       shape (3,)
            19:88   body_pose           shape (69,)
            88:98  betas                shape (10,)
        """

        c_global_orient = c[:, 16:19]
        c_body_pose = c[:, 19:88]
        c_betas = c[:, 88:98]

        with torch.no_grad():
             # Observation Space
            body_model_params = {'global_orient':c_global_orient, 'body_pose': c_body_pose, 'betas': c_betas}
            body_model_out = self.body_model(**body_model_params, return_verts=True)
            verts = body_model_out['vertices']
            del body_model_out

            canonical_mapping_kwargs = {
                'verts': verts,
            }
            return canonical_mapping_kwargs

    def get_part(self, image, c, canonical_mapping_kwargs):
        # Find head center and project to image.
        device = c.device
        B, C, H, W = image.shape
        # face size ratio set as 20/256
        box_size = (20/256) * self.img_resolution
        target_size = int(2*box_size)
        padding = False

        with torch.no_grad():
            verts = canonical_mapping_kwargs['verts']
            verts_canonical = self.canonical_smpl_info['verts_canonical']
            mask_front = (verts_canonical[0,:,1]>0.33) & (verts_canonical[0,:,2]>0)
            mask_back = (verts_canonical[0,:,1]>0.33) & (verts_canonical[0,:,2]<0)
            verts_front = verts[:,mask_front]
            verts_back = verts[:,mask_back]
            head_center_3D = ((verts_front.mean(1)+verts_back.mean(1))/2).unsqueeze(1)  # B,1,3

            if self.camera_type == 'blender':
                proj_camera = self.dmtet_geometry.renderer.blender_camera
            elif self.camera_type == 'smpl':
                proj_camera = self.dmtet_geometry.renderer.smpl_camera
            else:
                raise NotImplementedError

            cam_mv = c[:, :self.c_cam_dim].view(-1, 4, 4)
            head_center_3D = xfm_points(head_center_3D, cam_mv)
            head_center_2D = proj_camera.project(head_center_3D)  # Projection in the camera
            head_center_2D = head_center_2D[..., :3] / head_center_2D[..., -1:]
            head_center_2D = (head_center_2D + 1.0) * self.img_resolution * 0.5
            head_center_2D = head_center_2D[:, 0, :2]

            if torch.min(head_center_2D) < box_size:
                # pad image upper part with white
                padding = True
                head_center_2D[:, 1] += box_size
            bboxs = torch.cat([head_center_2D-box_size, head_center_2D+box_size], dim=-1)

        if padding:
            image = F.pad(image, (0,0,int(box_size),0), "constant", 1.0)

        image_part = roi_align(
            input=image.float(),
            boxes=torch.cat([torch.arange(B).view(B,1).float().to(device), bboxs.float() - 0.5], 1).float(),
            output_size=(target_size, target_size))
        return image_part.float()

    def generate_normal_map(
            self, ws_tex, c, ws_geo, return_eikonal=False, **block_kwargs):
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
            ws_geo = ws_geo[:, self.generator.tri_plane_synthesis.num_ws_geo:]
            canonical_mapping_kwargs = self.get_canonical_mapping_info(c)
            mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo, sdf_feature, canonical_mapping_kwargs)
        else:
            raise NotImplementedError
        # Render the mesh into 2D image (get 3d position of each image plane)
        # cam_mv: (batch_size, n_views, 4, 4)
        antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv)

        mask_pyramid = None

        surface_pos = return_value['tex_pos']
        
        surface_hard_mask = hard_mask
        surface_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in surface_pos]

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

        normal_img = normal_feat.permute(0, 3, 1, 2)
        img_buffers_viz = None

        final_img = torch.cat([normal_img, antilias_mask.permute(0, 3, 1, 2)], dim=1)

        if self.part_disc:
            part_img = self.get_part(final_img, c, canonical_mapping_kwargs)
        else:
            part_img = None
        
        return final_img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_buffers_viz, \
               mask_pyramid, surface_hard_mask, sdf_reg_loss, eikonal_loss, return_value, part_img

    def generate(
            self, ws_tex, c, ws_geo, return_eikonal=False, return_depth=False, **block_kwargs):
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
        antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv)

        mask_pyramid = None

        tex_pos = return_value['tex_pos']
        if return_depth:
            depth = return_value['depth']
            depth = torch.cat(depth).permute(0, 3, 1, 2)

        tex_hard_mask = hard_mask
        tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]

        
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

        # Predict the RGB color for each pixel (self.to_rgb is 1x1 convolution)
        if self.feat_channel > 3:
            # network_out = self.to_rgb(img_feat.permute(0, 3, 1, 2), ws[:, -1])
            # network_out = self.rgb_decoder(img_feat)
            network_out = img_feat
        else:
            network_out = img_feat.permute(0, 3, 1, 2)

        rgb_feat =  network_out.permute(0, 3, 1, 2)
        img_buffers_viz = None

        if self.render_type == 'neural_render':
            img = self.to_rgb(rgb_feat, ws_tex_sr[:, -1])
        else:
            raise NotImplementedError
        
        if self.with_sr:
            sr_image = self.superresolution(
                img, rgb_feat, ws_tex_sr, noise_mode='none')
        else:
            sr_image = img

        final_img = torch.cat([sr_image, antilias_mask.permute(0, 3, 1, 2)], dim=1)

        if self.part_disc:
            part_img = self.get_part(final_img, c, canonical_mapping_kwargs)
        else:
            part_img = None

        if return_depth:
            return final_img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_buffers_viz, \
               mask_pyramid, tex_hard_mask, sdf_reg_loss, eikonal_loss, return_value, part_img, depth
        else:
            return final_img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_buffers_viz, \
               mask_pyramid, tex_hard_mask, sdf_reg_loss, eikonal_loss, return_value, part_img

    def forward_normal(self, ws, c, ws_geo, return_shape=False, return_eikonal=False, **block_kwargs):
        normal_img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, mask_pyramid, \
        tex_hard_mask, sdf_reg_loss, eikonal_loss, render_return_value, part_img = self.generate_normal_map(ws, c, ws_geo, return_eikonal=return_eikonal, **block_kwargs)
        if return_shape:
            return normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value, part_img
        return normal_img, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value, part_img

    def forward(self, ws, c, ws_geo, return_shape=False, return_eikonal=False, **block_kwargs):
        img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, mask_pyramid, \
        tex_hard_mask, sdf_reg_loss, eikonal_loss, render_return_value, part_img = self.generate(ws, c, ws_geo, return_eikonal=return_eikonal, **block_kwargs)
        if return_shape:
            return img, sdf, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value, part_img
        return img, mask_pyramid, sdf_reg_loss, eikonal_loss, render_return_value, part_img


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
            self, z, geo_z, c, truncation_psi=1, 
            truncation_cutoff=None, update_emas=False, 
            # camera=None,
            generate_no_light=False,
            only_img=False,
            return_depth=False,
            ws=None, ws_geo=None,
            **synthesis_kwargs):

        c_cam_perm = torch.zeros_like(c[:, :self.c_cam_dim])

        """
        ws: b, ws, 512
        ws_geo: b, ws_geo, 512
        z: b, 512
        geo_z: b, 512
        """
        if ws is None and ws_geo is None:
            ws = self.mapping(
                z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            ws_geo = self.mapping_geo(
                geo_z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                update_emas=update_emas)

        if return_depth:
            img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, mask_pyramid, tex_hard_mask, \
            sdf_reg_loss, eikonal_loss, render_return_value, part_img, depth = self.synthesis.generate(
                ws, c=c,
                ws_geo=ws_geo,
                return_depth=True,
                **synthesis_kwargs)
        else:
            img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, mask_pyramid, tex_hard_mask, \
            sdf_reg_loss, eikonal_loss, render_return_value, part_img = self.synthesis.generate(
                ws, c=c,
                ws_geo=ws_geo,
                **synthesis_kwargs)

        if return_depth:
            _, _, fine_depth = self.generate_dense_mesh(ws=ws, ws_geo=ws_geo, z=None, geo_z=None, c=c, return_depth=True)
        
        ##############################
        if only_img:
            normal_img = normal_part_img = None
        else:
            normal_img, _, _, _, _, _, _, _, _, _, _, _, _, normal_part_img = self.synthesis.generate_normal_map(
                ws, c, 
                ws_geo=ws_geo,
                **synthesis_kwargs)
        ##############################
        if generate_no_light:
            if return_depth:
                return img, mask, normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, tex_hard_mask, part_img, normal_part_img, fine_depth
            else:
                return img, mask, normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, tex_hard_mask, part_img, normal_part_img
        if return_depth:
            return img, mask, normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, tex_hard_mask, part_img, normal_part_img, fine_depth
        else:
            return img, mask, normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, tex_hard_mask, part_img, normal_part_img

    def generate_dense_mesh(
            self, z, geo_z, c, ws=None, ws_geo=None, truncation_psi=1, 
            truncation_cutoff=None, update_emas=False, 
            generate_raw=False, with_texture=False, return_depth=False,
            **synthesis_kwargs):
        c_cam_perm = torch.zeros_like(c[:, :self.c_cam_dim])
        
        if ws is None and ws_geo is None:
            ws = self.mapping(
                z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            ws_geo = self.mapping_geo(
                geo_z, c_cam_perm, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                update_emas=update_emas)

        if not with_texture:
            if return_depth:
                mesh_v, mesh_f, depth = self.synthesis.extract_mesh_with_marching_cubes(
                    ws, c=c,
                    ws_geo=ws_geo,
                    with_texture=with_texture,
                    return_depth=True,
                    **synthesis_kwargs)
                return mesh_v, mesh_f, depth
            else:
                mesh_v, mesh_f = self.synthesis.extract_mesh_with_marching_cubes(
                    ws, c=c,
                    ws_geo=ws_geo,
                    with_texture=with_texture,
                    **synthesis_kwargs)
                return mesh_v, mesh_f
        else:
            if return_depth:
                mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out =self.synthesis.extract_mesh_with_marching_cubes(
                    ws, c=c,
                    ws_geo=ws_geo,
                    with_texture=with_texture,
                    **synthesis_kwargs)
                return mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out
            else:
                mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out, depth =self.synthesis.extract_mesh_with_marching_cubes(
                    ws, c=c,
                    ws_geo=ws_geo,
                    with_texture=with_texture,
                    return_depth=True,
                    **synthesis_kwargs)
                return mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out, depth

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

        img, sdf, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, _, _, _, _ = self.synthesis(
            ws=ws, 
            c=c,
            ws_geo=ws_geo,
            update_emas=update_emas,
            return_shape=True,
        )
        return img
