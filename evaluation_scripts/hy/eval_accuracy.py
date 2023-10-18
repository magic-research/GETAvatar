# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
from matplotlib.colors import NoNorm
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import mrcfile

import legacy

from camera_utils import LookAtPoseSampler
from torch_utils import misc
import glob
import numpy as np
import PIL.Image
from torch_utils.ops import upfirdn2d

import json
import sys
sys.path.append('../../../im3dmm')
from morphable_model.mpi_flame.FLAME import FLAME
sys.path.append('../../../../navi_dev/scripts')
import render_mesh_on_image
import argparse
import cv2
#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def generate_ds_images(G, psi=1, truncation_cutoff=14, cfg='FFHQ',device=torch.device('cuda'), label_pool='', outdir='', **video_kwargs):
    
    num_images = 10#250

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)

    feature_dim = G.rendering_kwargs.get('feature_dim', 0)
    if cfg == 'ffhq':
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    else:
        intrinsics = torch.tensor([[1015./224, 0.0, 0.5],[0.0, 1015./224, 0.5], [0.0, 0.0, 1.0]], device=device)

    frame_3dmms = np.load(label_pool)

    img_outdir = os.path.join(outdir, 'z_interp')
    if not os.path.exists(img_outdir):
        os.mkdir(img_outdir)


    # selected_seeds = [302, 317, 297]
    for frame_idx in tqdm(range(num_images)):
        param_seed = frame_idx * 4 + 5000
        frame_3dmm = frame_3dmms[param_seed, :]
        # print(frame_3dmm.shape)
        # np.save(os.path.join(outdir, 'seed%04d.npy'%frame_idx), frame_3dmm)
        frame_3dmm = torch.from_numpy(frame_3dmm).to(device).float()
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)
        
        num_frames = 100
        for sample_idx in range(num_frames):
            zs =  torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in [sample_idx * 0 + frame_idx * 10]])).to(device)
            np.random.seed(sample_idx)
            param_seed = np.random.randint(0, len(frame_3dmms))
            #param_seed = sample_idx * 4 + 5000
            frame_3dmm[..., 25:125] =torch.from_numpy(frame_3dmms[param_seed, 25:125]).to(device).float()

            # frame_3dmm = torch.from_numpy(frame_3dmm).to(device).float()
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)
      
            ws = G.mapping(zs, conditioning_params, truncation_psi=psi, truncation_cutoff=truncation_cutoff)

            #rendering_params = frame_3dmm.reshape(-1, feature_dim + 25)
            out = G.synthesis(ws, conditioning_params.repeat([len(ws), 1]), noise_mode='const')
            img = out['image'][0].permute(1, 2, 0)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            output_path = os.path.join(img_outdir, 'seed%04d_%04d.png' % (frame_idx, sample_idx))
            PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(output_path)

def generate_sample_images(G, psi=1, truncation_cutoff=14, cfg='FFHQ',device=torch.device('cuda'), label_pool='', outdir='', **video_kwargs):
    
    num_images = 1000

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)

    feature_dim = G.rendering_kwargs.get('feature_dim', 0)
    if cfg == 'ffhq':
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    else:
        intrinsics = torch.tensor([[1015./224, 0.0, 0.5],[0.0, 1015./224, 0.5], [0.0, 0.0, 1.0]], device=device)

    frame_3dmms = np.load(label_pool)

    img_outdir = os.path.join(outdir, 'group_samples')
    if not os.path.exists(img_outdir):
        os.mkdir(img_outdir)

    seeds_pool = [47,65,15, 8, 300,  55, 13,  59,  85, 303, 5, 6, 303, 13,558, 113, 119, 85]
    exp_seeds = [0,1,2,3,4,15, 6,17,12,9]
    # selected_seeds = [302, 317, 297]
    for frame_idx in tqdm(range(num_images)):
        # param_seed = frame_idx * 4 + 5000
        # np.random.seed(frame_idx+100)
        # param_seed = np.random.randint(0, len(frame_3dmms))
        frame_3dmm = frame_3dmms[frame_idx, :]
        # print(frame_3dmm.shape)
        # np.save(os.path.join(outdir, 'seed%04d.npy'%frame_idx), frame_3dmm)
        frame_3dmm = torch.from_numpy(frame_3dmm).to(device).float()
        # shape = torch.zeros_like(frame_3dmm[..., 25:125])
        shape = frame_3dmm[..., 25:125]
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)
        
        zs =  torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in [frame_idx]])).to(device)
        ws = G.mapping(zs, conditioning_params, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
        num_frames = 10
        for sample_idx in range(num_frames):
            # np.random.seed(exp_seeds[sample_idx])
            # exp_seed = np.random.randint(0, len(frame_3dmms))
            exp_seed = sample_idx + num_frames * frame_idx
            #param_seed = sample_idx * 4 + 5000
            new_frame_3dmm = torch.from_numpy(frame_3dmms[exp_seed, :]).to(device).float()
            new_frame_3dmm[..., 25:125] = shape
                
            pitch = 0.1 * np.random.uniform(-1, 1)
            yaw = 0.3 * np.random.uniform(-1, 1)
            cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw,
                                                3.14/2 -0.05 + pitch,
                                                cam_pivot, radius=cam_radius, device=device)
            # frame_3dmm = torch.from_numpy(frame_3dmm).to(device).float()
            conditioning_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9),new_frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)
            # conditioning_params = new_frame_3dmm.reshape(1, -1)

            new_zs =  torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in [exp_seed]])).to(device)
            
            ws_bcg = G.mapping(new_zs, conditioning_params, truncation_psi=psi, truncation_cutoff=truncation_cutoff)

            #rendering_params = frame_3dmm.reshape(-1, feature_dim + 25)
            out = G.synthesis(ws, conditioning_params.repeat([len(ws), 1]), ws_bcg=ws_bcg, noise_mode='const')
            img = out['image'][0].permute(1, 2, 0)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            output_path = os.path.join(img_outdir, 'seed%04d_%04d.png' % (frame_idx, sample_idx))
            PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(output_path)

def render_mesh(mesh, mesh_output_path='/tmp/test.ply', img_width=512, img_height=512, camera_position = None, camera_json=None):
    mesh.export(mesh_output_path)

    img_width = 512
    img_height = 512
    if camera_position is None or camera_json is None:
        vlist = mesh.vertices
        vlist_range = np.amax(vlist, axis=0) - np.amin(vlist, axis=0)
        vctr = np.average(vlist, axis=0)
        camera_position = vctr
        z_offset = 2.0 * vlist_range[2]
        camera_position[2] += z_offset
        camera_json = render_mesh_on_image.create_camera_intrinsics_json(
            img_width, img_height, focal_length=img_width)
    img = render_mesh_on_image.render_untextured_mesh(mesh_output_path, img_width, img_height, [1.0, 1.0, 1.0], camera_position, camera_json)

    return img, camera_position, camera_json

def generate_anim(G, psi=1, truncation_cutoff=14, cfg='FFHQ',device=torch.device('cuda'), label_pool='', outdir='', **video_kwargs):
    
    # num_images = 100
    # seeds = [560, 510, 486, 477, 474, 451,324,317,300, 297,246,230,176,169,93,65,13]
    # seeds = [317, 300, 297, 302, 2, 317]
    seeds=[317]

    face_model_config = argparse.ArgumentParser()
    with open('~/Code/morphable_nerf/im3dmm/morphable_model/mpi_flame/flame_config.json', 'r') as f:
        face_model_config.__dict__ = json.load(f)
    flame_model= FLAME(face_model_config).cuda(device)

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)

    feature_dim = G.rendering_kwargs.get('feature_dim', 0)
    if cfg == 'ffhq':
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    else:
        intrinsics = torch.tensor([[1015./224, 0.0, 0.5],[0.0, 1015./224, 0.5], [0.0, 0.0, 1.0]], device=device)

    frame_3dmms = np.load('~/Data/neural_rendering/eg3d/dataset/ffhq_labels.npy')
    label_paths = glob.glob(os.path.join(label_pool, '*.npy'))
    label_paths.sort()
    
    anim_data = []
    n_interp = 30
    # label_paths = label_paths[0:1200][::n_interp]
    # num_frames = len(label_paths)
    # for path in label_paths:
    #     data = np.load(path)
    #     anim_data.append(data)
    # anim_data = np.stack(anim_data, axis=0)
    num_frames = 8
    anim_data = np.stack([frame_3dmms[8, 25:], frame_3dmms[80, 25:], 
        frame_3dmms[68268, 25:], frame_3dmms[89256, 25:], frame_3dmms[71530, 25:], frame_3dmms[86364, 25:], 
        frame_3dmms[117355, 25:], frame_3dmms[98642, 25:]], axis = 0)
    frame_indices = np.arange(num_frames) * n_interp
    interp_func = scipy.interpolate.interp1d(frame_indices, anim_data, axis = 0)

    img_outdir = os.path.join(outdir, 'exp_interp_shape')
    if not os.path.exists(img_outdir):
        os.mkdir(img_outdir)

    frame_3dmm = frame_3dmms[seeds[0]:seeds[0]+1, :]
    frame_3dmm[..., 25:125] = 0
    exp = frame_3dmm[..., 125:]
    
    frame_3dmm = torch.from_numpy(frame_3dmm).to(device).float()
            
    selected_shape_indices=[-1]
    for rand_seed in [ 80, 97, 76, 16, 94]:
        np.random.seed(rand_seed)
        shape_idx = np.random.randint(0, len(frame_3dmms))
        selected_shape_indices.append(shape_idx)
    selected_shape_indices.append(-1)
        
    selected_exp_indices=[-1]
    for rand_seed in [1,2,3,4,6,7,8,9,16,19]:
        np.random.seed(rand_seed)
        exp_idx = np.random.randint(0, len(frame_3dmms))
        frame_3dmms[exp_idx, -3:] += np.random.normal(0, 0.12, 3)
        selected_exp_indices.append(exp_idx)
    selected_exp_indices.append(-1)

    total_frames = (num_frames - 1) * n_interp
    ws_bcg = None
    cam_json = None
    cam_pos = None
    for frame_idx in tqdm(range(len(seeds))):
        shape = frame_3dmm[..., 25:125]
        zs =  torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in [seeds[frame_idx]]])).to(device)

        num_substeps = 1
        total_frames = num_substeps * len(selected_exp_indices)
                    
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)
        ws = G.mapping(zs, conditioning_params, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
        for sample_idx in range(total_frames-num_substeps):  
            current_exp_idx = selected_exp_indices[ sample_idx // num_substeps]
            next_exp_idx = selected_exp_indices[ sample_idx // num_substeps + 1]
            if current_exp_idx != -1:
                current_exp = frame_3dmms[current_exp_idx:current_exp_idx+1, 125:]
            else:
                current_exp = exp

            if next_exp_idx != -1:    
                next_exp = frame_3dmms[next_exp_idx:next_exp_idx+1, 125:]
            else:
                next_exp = exp

            w = (sample_idx % num_substeps) / num_substeps
            interp_exp = torch.from_numpy((1-w) * current_exp + w * next_exp).to(device).float()
            frame_3dmm[..., 125:] = interp_exp
            frame_3dmm[..., 25:125] = torch.from_numpy(np.random.normal(0, 1, size=[1, 100])).to(device).float()
            
            shape, expression, jaw_pose, neck_pose = frame_3dmm[..., -206:].split([100, 100, 3, 3], dim=-1)
            pose_params = torch.cat([torch.zeros_like(jaw_pose), jaw_pose], dim =-1)
            verts, _ = flame_model(shape_params=shape, expression_params=expression, pose_params=pose_params, neck_pose=neck_pose)
            import trimesh
            mesh = trimesh.base.Trimesh(vertices = verts[0].detach().cpu().numpy(), faces=flame_model.faces)
            mesh_path = os.path.join(img_outdir, 'flame%04d.obj'%sample_idx)
            mesh.export(mesh_path)
            img, cam_pos, cam_json = render_mesh(mesh, mesh_path, camera_position=cam_pos, camera_json=cam_json)
            img_output_path = os.path.join(img_outdir, 'flame%04d.png'%sample_idx)
            cv2.imwrite(img_output_path, img)
            
            if frame_idx == 0 and sample_idx == 0:
                ws_bcg = ws.clone()

            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)
            imgs = []
            angle_p = -0.05
        
            # for angle_y, angle_p in [(0.6 - 0.015 * seed_idx, angle_p)]:
            for angle_y, angle_p in [(0, angle_p)]:
                # pitch_range = 0.2 
                # yaw_range = 0.5 
                # cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * min(sample_idx, total_frames-1) / (total_frames-1)),
                #                                 3.14/2 -0.05 + pitch_range * np.sin(2 * 3.14 * min(sample_idx, total_frames-1)/ (total_frames-1)),
                #                                 cam_pivot, radius=cam_radius, device=device)
                
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)

                out = G.synthesis(ws, camera_params.repeat([len(ws), 1]), noise_mode='const', ws_bcg=ws_bcg)
                img = out['image'][0].permute(1, 2, 0)
                img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs.append(img)
            output_path = os.path.join(img_outdir, 'seed%04d_%04d_%04d.png' % (frame_idx, sample_idx, 0))
            img = torch.cat(imgs, dim=1)
            PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(output_path)

def generate_ds_anims(G, psi=1, truncation_cutoff=14, cfg='FFHQ',device=torch.device('cuda'), label_pool='', outdir='', **video_kwargs):
    seeds = [169, 451, 477, 176]

    face_model_config = argparse.ArgumentParser()
    with open('~/Code/morphable_nerf/im3dmm/morphable_model/mpi_flame/flame_config.json', 'r') as f:
        face_model_config.__dict__ = json.load(f)
    flame_model= FLAME(face_model_config).cuda(device)

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)

    feature_dim = G.rendering_kwargs.get('feature_dim', 0)
    if cfg == 'ffhq':
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    else:
        intrinsics = torch.tensor([[1015./224, 0.0, 0.5],[0.0, 1015./224, 0.5], [0.0, 0.0, 1.0]], device=device)

    frame_3dmms = np.load(label_pool)

    img_outdir = os.path.join(outdir, 'geom')
    if not os.path.exists(img_outdir):
        os.mkdir(img_outdir)

    for frame_idx in tqdm(range(len(seeds))):
        # param_seed = frame_idx * 4 + 5000
    
        frame_3dmm = frame_3dmms[frame_idx:frame_idx+1, :]
        shape = frame_3dmm[..., 25:125]
        frame_3dmm = torch.from_numpy(frame_3dmm).to(device).float()
        zs =  torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in [seeds[frame_idx]]])).to(device)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)
        ws = G.mapping(zs, conditioning_params, truncation_psi=psi, truncation_cutoff=truncation_cutoff)

        for sample_idx in range(1):
            np.random.seed(189)
            frame_seed = np.random.randint(0, len(frame_3dmms))
            # print (frame_seed)
            frame_3dmm = frame_3dmms[frame_seed:frame_seed+1, :]
            frame_3dmm[..., 25:125] = shape
            # frame_3dmm[..., -3:] += np.random.
            # print(frame_3dmm.shape)
            frame_3dmm = torch.from_numpy(frame_3dmm).to(device).float()
            torch.manual_seed(sample_idx)
            neck_pose = torch.normal(mean=torch.zeros(1, 3), std=0.04).cuda(device).float()
            frame_3dmm[..., -3:] = neck_pose

            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)
            imgs = []
            angle_p = -0.05
        
            # for angle_y, angle_p in [(0.6 - 0.015 * seed_idx, angle_p)]:
            for angle_y, angle_p in [(.3, angle_p), (0, angle_p), (-.3, angle_p)]:
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), frame_3dmm[...,-feature_dim:].reshape(-1, feature_dim)], 1)

                out = G.synthesis(ws, camera_params.repeat([len(ws), 1]), noise_mode='const')
                img = out['image'][0].permute(1, 2, 0)
                img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs.append(img)
            output_path = os.path.join(img_outdir, 'seed%04d_%04d.png' % (seeds[frame_idx], sample_idx))
            img = torch.cat(imgs, dim=1)
            PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(output_path)

            shape, expression, jaw_pose, neck_pose = frame_3dmm[..., -206:].split([100, 100, 3, 3], dim=-1)
            pose_params = torch.cat([torch.zeros_like(jaw_pose), jaw_pose], dim =-1)
            verts, _ = flame_model(shape_params=shape, expression_params=expression, pose_params=pose_params, neck_pose=neck_pose)
            import trimesh
            mesh = trimesh.base.Trimesh(vertices = verts[0].detach().cpu().numpy(), faces=flame_model.faces)
            mesh_path = os.path.join(img_outdir, f'seed%04d_%04d_flame.obj' % (seeds[frame_idx], sample_idx))
            mesh.export(mesh_path)

            shape = True
            shape_prior = True
            shape_format = '.ply'
            if shape:
                # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
                max_batch=500000
                shape_res = 512
                samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'][2] * 1.2)#.reshape(1, -1, 3)
                samples = samples.to(device)
                shape_sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
                transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
                transformed_ray_directions_expanded[..., -1] = -1

                if shape_prior:
                    prior_sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
                else:
                    prior_sigmas = None
                head = 0
                with tqdm(total = samples.shape[1]) as pbar:
                    with torch.no_grad():
                        while head < samples.shape[1]:
                            torch.manual_seed(0)
                            out = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], zs, conditioning_params, truncation_psi=psi, truncation_cutoff=truncation_cutoff, noise_mode='const', return_weighted_geom_delta=True)
                            if 'sdf' in out:
                                sigma = out['sdf']
                            else:
                                sigma = out['sigma']
                            shape_sigmas[:, head:head+max_batch] = sigma
                            if shape_prior:
                                prior_sigmas[:, head:head+max_batch] = sigma - out['geom_delta']
                            head += max_batch
                            pbar.update(max_batch)

            for sigmas, name in [(shape_sigmas, 'final'), (prior_sigmas, 'prior')]:
                if sigmas is None:
                    continue
                sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
                sigmas = np.flip(sigmas, 0)

                # Trim the border of the extracted cube
                pad = int(30 * shape_res / 256)
                if G.rendering_kwargs['with_sdf']:
                    pad_value = 1.0
                else:
                    pad_value = -1000
                sigmas[:pad] = pad_value
                sigmas[-pad:] = pad_value
                sigmas[:, :pad] = pad_value
                sigmas[:, -pad:] = pad_value
                sigmas[:, :, :pad] = pad_value
                sigmas[:, :, -pad:] = pad_value

                if shape_format == '.ply':
                    from shape_utils import convert_sdf_samples_to_ply
                    if G.rendering_kwargs['with_sdf']:
                        level = 0
                    elif name == 'prior':
                        level = 500
                    else:
                        level = 10
                    convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [-0.5, -0.5, -0.5], 1./shape_res, os.path.join(img_outdir, 'seed%04d_%04d_%s.ply' % (seeds[frame_idx], sample_idx, name)), level=level)
#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--cfg', help='Config', type=click.Choice(['ffhq', 'cats', 'ffhq_3dmm']), required=False, metavar='STR', default='ffhq', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']), required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=1, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)
@click.option('--interpolate', type=bool, help='Interpolate between seeds', default=True, show_default=True)
@click.option('--label-pool', help='label pool', type=str, required=False, default='')
@click.option('--fix-z', help='Export geometric prior?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--fix-p', help='Export geometric prior?', type=bool, required=False, metavar='BOOL', default=True, show_default=True)



def generate_images(
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    outdir: str,
    reload_modules: bool,
    cfg: str,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    shapes: bool,
    interpolate: bool,
    label_pool: str,
    fix_z: bool,
    fix_p: bool,
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device) # type: ignore
        # D = network['D'].to(device)
        D = None


    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    G.rendering_kwargs['det_sampling'] = True
    G.decoder.pose_perturb_magtinude = 0
    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    #generate_paired_images(G=G, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, label_pool=label_pool, outdir=outdir)
    #generate_ds_images(G=G, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, label_pool=label_pool, outdir=outdir)
    generate_sample_images(G=G, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, label_pool=label_pool, outdir=outdir)
    # generate_ds_anims(G=G, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, label_pool=label_pool, outdir=outdir)
    # generate_anim(G=G, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, label_pool=label_pool, outdir=outdir)

def eval_accuracy():
    fitting_paths = glob.glob('~/Data/neural_rendering/eg3d/exp/accuracy/fittings/*npy')
    fitting_paths.sort()

    face_model_config = argparse.ArgumentParser()
    with open('~/Code/morphable_nerf/im3dmm/morphable_model/mpi_flame/flame_config.json', 'r') as f:
        face_model_config.__dict__ = json.load(f)
    device = 'cuda'
    flame_model= FLAME(face_model_config).cuda(device)

    shape_errors = []
    exp_errors = []
    pose_errors = []
    shape_params_errors = []
    exp_params_errors = []

    img_paths = glob.glob('~/Data/neural_rendering/eg3d/exp/accuracy_nogeomprior/cropped_images/seed0142*png')
    output_dir = '~/Data/neural_rendering/eg3d/exp/accuracy_nogeomprior/overlay'
    img_paths.sort()
    from skimage import io, data, draw
    lmks = np.load('~/Data/neural_rendering/eg3d/exp/accuracy/mean_lmks.npy')
    for img_path in img_paths:
        input = io.imread(img_path)[...,:3]
        for k, lmk in enumerate(lmks):
            r = 6
            row, col = draw.disk(lmk[:2], r)
            conf = 0
            color = conf * np.array([0, 255, 0]) + (1.0 - conf) * np.array([255, 0, 0])

            input[col, row, :] = np.array(color, np.int64)

        basename = os.path.basename(img_path)    
        io.imsave(os.path.join(output_dir, basename), input) 

    diff = []
    # lmks_2d_paths = glob.glob('/home/tiger/Data/neural_rendering/eg3d/exp/accuracy/landmarks/*npy')
    # lmks_2d_paths.sort()
    # lmks_2d_errors = []
    # for pid in range(0, len(lmks_2d_paths), 10):
    #     datas = []
    #     for i in range(10):
    #         data = np.load(lmks_2d_paths[pid+i])
    #         datas.append(data)
    #     mean_data = np.array(datas).mean(0)
    #     lmks_2d_errors.append((datas - mean_data).abs().mean())
        
    # for path in lmks_2d_paths:
    #     lmks = np.load(path)

    for pid, path in enumerate(fitting_paths):
        fitting = np.load(path)
        gt_path = '~/Data/neural_rendering/eg3d/exp/accuracy/input_params/seed%04d.npy' % (pid // 10)
        gt_data = np.load(gt_path)

        gt_params = torch.from_numpy(gt_data).to(device).float()              
        gt_shape, gt_expression, gt_jaw_pose, gt_neck_pose = gt_params[None, -206:].split([100, 100, 3, 3], dim=-1)
        gt_pose_params = torch.cat([torch.zeros_like(gt_jaw_pose), gt_jaw_pose], dim =-1)
        
        # if pid == 2100 or pid  == 680 or pid == 1740:
        #     print('here')
        fit_params = torch.from_numpy(fitting).to(device).float()              
        fit_shape, fit_expression, fit_jaw_pose, fit_neck_pose = fit_params[None, -206:].split([100, 100, 3, 3], dim=-1)
        fit_pose_params = torch.cat([torch.zeros_like(fit_jaw_pose), fit_jaw_pose], dim =-1)

        gt_verts, _ = flame_model(shape_params=gt_shape, expression_params=torch.zeros_like(gt_expression), pose_params=torch.zeros_like(gt_pose_params), neck_pose=torch.zeros_like(gt_neck_pose))
        fit_verts, _ = flame_model(shape_params=fit_shape, expression_params=torch.zeros_like(gt_expression), pose_params=torch.zeros_like(gt_pose_params), neck_pose=torch.zeros_like(gt_neck_pose))        

        # if pid // 10 == 174:
        #     diff.append(fit_verts)
        shape_errors.append(np.abs((gt_verts.cpu().numpy() - fit_verts.cpu().numpy() )).mean())
                
        gt_verts, gt_lmks = flame_model(shape_params=torch.zeros_like(gt_shape), expression_params=gt_expression, pose_params=gt_pose_params, neck_pose=gt_neck_pose)
        fit_verts, fit_lmks = flame_model(shape_params=torch.zeros_like(gt_shape), expression_params=fit_expression, pose_params=fit_pose_params, neck_pose=fit_neck_pose)       

        exp_errors.append(np.abs((gt_lmks.cpu().numpy()  - fit_lmks.cpu().numpy() )).mean())
        pose_errors.append(np.abs((fitting[..., :16] - gt_data[..., :16])).mean())
        shape_params_errors.append(np.abs((fit_shape.cpu().numpy()  - gt_shape.cpu().numpy())).mean())
        exp_params_errors.append(np.abs((gt_params[..., -106:].cpu().numpy()  - fit_params[..., -106:].cpu().numpy())).mean())
    shape_errors = np.array(shape_errors).reshape(-1, 10)
    exp_errors = np.array(exp_errors).reshape(-1, 10)
    pose_errors = np.array(pose_errors).reshape(-1, 10)
    shape_params_errors = np.array(shape_params_errors).reshape(-1, 10)
    exp_params_errors = np.array(exp_params_errors).reshape(-1, 10)    

    # a = np.max(shape_errors, -1)
    # b = np.min(shape_errors, -1)
    # for k in diff:
    #     print (torch.abs((k - diff[8])).mean()) 
    print ('num_samples: %d | shape error: %f | exp error: %f | pose_error: %f | shape dev: %f | exp dev: %f | pose dev: %f' % 
        (len(shape_errors), shape_errors.mean(), exp_errors.mean(), pose_errors.mean(), 
         np.var(shape_params_errors, -1).mean(), np.var(exp_params_errors, -1).mean(), np.var(pose_errors, -1).mean()))

if __name__ == "__main__":
    # paths = glob.glob('/home/tiger/Data/neural_rendering/eg3d/exp/ds_pose/*npy')
    # paths.sort()
    # shapes=[]
    # exps=[]
    # cams=[]
    # for path in paths:
    #     data = np.load(path)
    #     shape = data[:100]
    #     exp = np.concatenate([data[100:150], data[206:]], axis=-1)
    #     cam = np.concatenate([data[200:203], data[203:204]/10. - 1., data[204:206]/10.], axis=-1)
    #     shapes.append(shape)
    #     exps.append(exp)
    #     cams.append(cam)
    # shapes = np.array(shapes)
    # cams = np.array(cams)
    # exps = np.array(exps)
    # print ('variance of shape: %f | variance of cam: %f | variance of exps: %f' % (np.var(shapes), np.var(cams), np.var(exps)))
    # var_shape = np.var(shapes)
    # var_cam = np.var(cams)
    # var_exp = np.var(exps)
    # print ('ds of shape: %f | ds of cam: %f | ds of exps: %f' % (
    #     (var_shape/var_cam) * (var_shape/var_exp), 
    #     (var_cam/var_shape) * (var_cam/var_exp),  
    #     (var_exp/var_shape) * (var_exp/var_cam)))
    generate_images() # pylint: disable=no-value-for-parameter
    # eval_accuracy()

#----------------------------------------------------------------------------
