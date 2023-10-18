# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import copy
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.interpolate
from PIL import Image

import torch
from tqdm import tqdm
import math
import click
import dnnlib
import imageio
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

from camera_utils import LookAtPoseSampler
# from metrics import metric_main
# from training.inference_utils import save_visualization, save_visualization_for_interpolation, \
#     save_textured_mesh_for_inference, save_geo_for_inference


#------------------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------------------
def gen_interp_video(
        G, 
        dataset, 
        save_gif, mp4: str, 
        seeds, 
        shuffle_seed=None, 
        w_frames=60*4, 
        kind='cubic', 
        grid_dims=(1,1), 
        num_keyframes=None, 
        wraps=2, 
        psi=1, 
        truncation_cutoff=14, 
        render_all_pose=False,
        gen_shapes=False, 
        device=torch.device('cuda'), 
        **video_kwargs
):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    if dataset is not None:
        labels = dataset._get_raw_labels()
        rand_idx = np.random.RandomState(seed=0).choice(range(len(labels)), len(all_seeds))
        cs = torch.from_numpy(labels[rand_idx]).to(device)
        # pose_template = get_canonical_pose()
    else:
        print('error')
    
    c_cam_dim = 16
    # fix camera to fix color
    c_cam = cs[:, :c_cam_dim].clone()
    # fix smpl
    c_smpl_input = cs[:, c_cam_dim:].clone() 
    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    geo_z = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)

    c_cam_perm = torch.zeros_like(cs[:, :c_cam_dim])
    ws = G.mapping(zs, c_cam_perm, truncation_psi=psi)
    ws_geo = G.mapping_geo(geo_z, c_cam_perm, truncation_psi=psi)

    # _ = G.synthesis(ws[:1], ws_geo[:1], c=cs[:1]) # warm up
    zs = zs.reshape(grid_h, grid_w, num_keyframes, *zs.shape[1:])
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
    ws_geo = ws_geo.reshape(grid_h, grid_w, num_keyframes, *ws_geo.shape[1:])
    c_smpl_input = c_smpl_input.reshape(grid_h, grid_w, num_keyframes, *c_smpl_input.shape[1:])

    # Interpolation.
    grid_tex = []
    grid_geo = []
    grid_smpl = []
    for yi in range(grid_h):
        row_geo = []
        row_tex = []
        row_smpl = []
        for xi in range(grid_w):
            # wrap: repeat the ws for 2*wrap+1 times for stablility?
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y_geo = np.tile(ws_geo[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            y_tex = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            y_smpl = np.tile(c_smpl_input[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp_geo = scipy.interpolate.interp1d(x, y_geo, kind=kind, axis=0)
            interp_tex = scipy.interpolate.interp1d(x, y_tex, kind=kind, axis=0)
            interp_smpl = scipy.interpolate.interp1d(x, y_smpl, kind=kind, axis=0)
            row_geo.append(interp_geo)
            row_tex.append(interp_tex)
            row_smpl.append(interp_smpl)
        grid_geo.append(row_geo)
        grid_tex.append(row_tex)
        grid_smpl.append(row_smpl)

    # Render video.
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    img_frames = []

    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        t = frame_idx / (num_keyframes * w_frames)
        azim = math.pi * 1 * np.cos(t * 1 * math.pi)
        elev = math.pi / 2

        cam2world_matrix = LookAtPoseSampler.sample(azim, elev, torch.tensor([0, -0.3, 0], device=device), radius=2.3, device=device)
        # TODO: check the matrix shape
        world2cam_matrix = torch.linalg.inv(cam2world_matrix)
        c = world2cam_matrix.reshape(-1, 16)

        for yi in range(grid_h):
            for xi in range(grid_w):
                all_imgs = []
                if render_all_pose:
                    pose_num = 4
                else:
                    pose_num = 1
                for idx in range(int(pose_num)):
                    # EG3D
                    interp_geo = grid_geo[yi][xi]
                    interp_tex = grid_tex[yi][xi]
                    interp_smpl = grid_smpl[yi][xi]
                    w_geo = torch.from_numpy(interp_geo(frame_idx / w_frames)).to(device).float()
                    w_tex = torch.from_numpy(interp_tex(frame_idx / w_frames)).to(device).float()
                    c_smpl = torch.from_numpy(interp_smpl(frame_idx / w_frames)).to(device).float()

                    if idx==1:
                        # Define T-pose
                        T_pose = torch.from_numpy(np.array([ 
                          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                          0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                          0.0000,  0.0000,  0.0000,  0.0000,  0.0000])).float()
                        c_smpl[:,3:72] = T_pose
                        # align global orient
                        c_smpl[:,0:3] = torch.from_numpy(np.array([0.0000,  0.0000,  0.0000])).float()
                    elif idx==2:    
                    # Define Da-pose
                        Da_pose = torch.from_numpy(np.array([ 
                            0.0000,  0.0000,  0.5000,  0.0000,  0.0000, -0.5000,  0.0000,  0.0000,
                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
                            0.0000,  0.0000,  0.0000,  0.0000,  0.0000])).float()
                        c_smpl[:,3:72] = Da_pose
                        # align global orient
                        c_smpl[:,0:3] = torch.from_numpy(np.array([0.0000,  0.0000,  0.0000])).float()
                    elif idx==3:
                        A_pose = torch.from_numpy(np.array([ 
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])).float()
                        c_smpl[:,3:72] = A_pose
                        # align global orient
                        c_smpl[:,0:3] = torch.from_numpy(np.array([0.0000,  0.0000,  0.0000])).float()
                    else:
                        pass

                    c_input = torch.cat((c, c_smpl.reshape(1, -1)), dim=-1).reshape(1, -1).float()

                    # final_img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_buffers_viz, \
                    # mask_pyramid, tex_hard_mask, sdf_reg_loss, eikonal_loss, return_value
                    # img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, img_buffers_viz, mask_pyramid, \
                    # tex_hard_mask, sdf_reg_loss, eikonal_loss, render_return_value = G.synthesis.generate(
                    #     ws_tex=w_tex.unsqueeze(0), c=c_input, ws_geo=w_geo.unsqueeze(0), 
                    #     noise_mode='const', truncation_psi=0.7)

                    img, _, _, _, _, _, _, _, _, _, _, _, _ = G.synthesis.generate(
                        ws_tex=w_tex.unsqueeze(0), c=c_input, ws_geo=w_geo.unsqueeze(0), 
                        noise_mode='const', truncation_psi=0.7)

                    G.requires_grad_(True)
                    normal_img, _, _, _, _, _, _, _, _, _, _, _, _ = G.synthesis.generate_normal_map(
                        ws_tex=w_tex.unsqueeze(0), c=c_input, ws_geo=w_geo.unsqueeze(0),
                        noise_mode='const', truncation_psi=0.7)
                    G.requires_grad_(False)
                    
                    rgb_img = img[:, :3] 
                    # all_imgs.append(rgb_img)
                    total_img = torch.cat((rgb_img, normal_img[:,:3]), dim=-1)
                    all_imgs.append(total_img)

                if len(all_imgs) > 1:
                    all_imgs = torch.cat(all_imgs, dim=-2)
                else:
                    all_imgs = all_imgs[0]

                imgs.append(all_imgs[0])

        img_grid = layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h)
        video_out.append_data(img_grid)
        if save_gif:
            img_frame = Image.fromarray(img_grid)
            img_frames.append(img_frame)
    video_out.close()
    if save_gif:
        img_frame.save(mp4.replace('.mp4', '.gif'), save_all=True, append_images=img_frames, duration=40, loop=0)

    return
# ----------------------------------------------------------------------------
def inference(
        seeds: List[int],
        output: str,
        shuffle_seed: Optional[int],
        grid: Tuple[int,int],
        num_keyframes: Optional[int],
        w_frames: int,
        truncation_psi: float,
        truncation_cutoff: int,
        save_gif: bool,
        # white_bg: bool,
        render_all_pose: bool,
        run_dir='.',  # Output directory.
        training_set_kwargs={},  # Options for training set.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus[.
        resume_pretrain=None,
        **dummy_kawargs
):
    from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import filtered_lrelu
    upfirdn2d._init()
    bias_act._init()
    filtered_lrelu._init()

    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = True  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = True  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.


    common_kwargs = dict(
        c_dim=0, img_resolution=training_set_kwargs['resolution'] if 'resolution' in training_set_kwargs else 1024, img_channels=3)
    G_kwargs['device'] = device

    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module

    # G_ema = copy.deepcopy(G).eval()  # deepcopy can make sure they are correct.
    G_ema = copy.deepcopy(G).eval()  # deepcopy can make sure they are correct.
    if resume_pretrain is not None and (rank == 0):
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G.load_state_dict(model_state_dict['G'], strict=True)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
    
    dataset = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset

    output_dir = os.path.dirname(output)
    os.makedirs(output_dir, exist_ok=True)

    gen_interp_video(   G=G_ema, 
                        dataset=dataset, 
                        save_gif=save_gif, 
                        mp4=output, 
                        bitrate='10M', 
                        grid_dims=grid, 
                        num_keyframes=num_keyframes, 
                        w_frames=w_frames, 
                        seeds=seeds, 
                        shuffle_seed=shuffle_seed, 
                        psi=truncation_psi, 
                        truncation_cutoff=truncation_cutoff,
                        render_all_pose=render_all_pose
                    )     
