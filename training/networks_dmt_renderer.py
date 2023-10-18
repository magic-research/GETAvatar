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
import trimesh
import torch.nn.functional as F
from torch_utils import persistence
import nvdiffrast.torch as dr
from training.sample_camera_distribution import sample_camera, create_camera_from_angle
from uni_rep.rep_3d.dmtet import DMTetGeometry
from uni_rep.camera.perspective_camera import PerspectiveCamera
from uni_rep.render.neural_render import NeuralRender
from training.discriminator_architecture import Discriminator
from training.geometry_predictor import Conv3DImplicitSynthesisNetwork, TriPlaneTex, \
    MappingNetwork, ToRGBLayer, TriPlaneTexGeo


@persistence.persistent_class
class DMTETRenderNetwork(torch.nn.Module):
    def __init__(
            self,
            img_resolution,  # Output image resolution.
            device='cuda',
            geometry_type='normal',
            tet_res=64,  # Resolution for tetrahedron grid
            render_type='neural_render',  # neural type
            deformation_multiplier=2.0,
            dmtet_scale=1.8,
            inference_noise_mode='random',
            **block_kwargs,  # Arguments for SynthesisBlock.
    ):  #
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.device = device
        # self.one_3d_generator = one_3d_generator
        self.inference_noise_mode = inference_noise_mode
        self.dmtet_scale = dmtet_scale
        self.deformation_multiplier = deformation_multiplier
        self.geometry_type = geometry_type

        self.render_type = render_type

        self.img_resolution = img_resolution
        self.grid_res = tet_res

        # Camera defination, we follow the defination from Blender (check the render_shapenet_data/rener_shapenet.py for more details)
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=self.device)
        # Renderer we used.
        dmtet_renderer = NeuralRender(device, camera_model=dmtet_camera)
        # Geometry class for DMTet
        self.dmtet_geometry = DMTetGeometry(
            grid_res=self.grid_res, scale=self.dmtet_scale, renderer=dmtet_renderer, render_type=render_type,
            device=self.device)


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

    def generate(self):
        scan_path = '/opt/tiger/jzihang/dataset/Renderpeople/rgb_dataset/rp_sample_2022-11-29/rigged_maya_model_split_1_of_5/rp_aaron_rigged_001_MAYA_a/mesh.obj'
        mesh = trimesh.load(scan_path)

        scan_faces = torch.tensor(np.array(mesh.faces.astype(np.int)), dtype=torch.int).to(self.device)
        scan_verts = torch.tensor(np.array(mesh.vertices.astype(np.float32))).to(self.device)

        # blender_transform_mtx = torch.tensor(
        #     [
        #         [1.0, 0.0, 0.0, 0.0],
        #         [0.0, -0.1432739794254303, -0.9896830320358276, -2.299999952316284],
        #         [0.0, 0.9896830916404724, -0.1432739645242691, -0.6659305691719055],
        #         [0.0, 0.0, 0.0, 1.0]
        #     ]
        # ).float().to(self.device)

        blender_transform_mtx = torch.tensor(
            [
                [0.39264795184135437, 0.17606116831302643, 0.9026793837547302, 2.0978055000305176],
                [0.9196889400482178, -0.07516677677631378, -0.3853859305381775, -0.8956277966499329],
                [0.0, 0.9815051555633545, -0.1914355605840683, -0.7778570055961609],
                [0.0, 0.0, 0.0, 1.0]
            ]
        ).float().to(self.device)

        re_correc_mtx = torch.tensor(
            [
                [1.0, 0.0, 0.0,  0.0],
                [0.0, 0.0, 1.0,  0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0,  1.0]
            ]
        ).float().to(self.device)

        world2cam_mtx = torch.linalg.inv(re_correc_mtx @ blender_transform_mtx).unsqueeze(0)

        mesh_v = [scan_verts]
        mesh_f = [scan_faces]
        cam_mv = [world2cam_mtx]
        
        # Render the mesh into 2D image (get 3d position of each image plane)
        antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv)
        # Merge them together
        print('antilias_mask.shape ', antilias_mask.shape)
        print('hard_mask.shape ', hard_mask.shape)

        return antilias_mask, hard_mask

    def forward(self):
        return 

