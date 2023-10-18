# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import copy
import os
import pickle
from time import perf_counter
import imageio
import skimage.io
import PIL.Image
from camera_utils import LookAtPoseSampler
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
import dnnlib
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from metrics import metric_main
from training.inference_utils import save_visualization, save_visualization_for_interpolation, \
    save_textured_mesh_for_inference, save_geo_for_inference, save_visualization_inference, save_visualization_inference_for_pck_eval, \
    save_visualization_inference_for_depth_eval


def clean_training_set_kwargs_for_metrics(training_set_kwargs):
    if 'add_camera_cond' in training_set_kwargs:
        training_set_kwargs['add_camera_cond'] = True
    return training_set_kwargs


# ----------------------------------------------------------------------------
def inversion(
        run_dir='.',  # Output directory.
        training_set_kwargs={},  # Options for training set.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus[.
        inference_vis=False,
        inference_for_depth_eval=False,
        inference_for_pck_eval=False,
        inference_to_generate_textured_mesh=False,
        resume_pretrain=None,
        inference_save_interpolation=False,
        inference_compute_fid=False,
        inference_generate_geo=False,
        num_steps=100,
        num_steps_pti=0,
        save_video=False,
        save_multiview=False,
        save_animation=False,
        use_normal_map=False,
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
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).requires_grad_(False).to(device)  # subclass of torch.nn.Module
    if resume_pretrain is not None and (rank == 0):
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G.load_state_dict(model_state_dict['G'], strict=True)
    dataset = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset

    print('==> inversion')
    os.makedirs(run_dir, exist_ok=True)
    run_list = [500, 10, 11, 12, 38, 142, 186, 193, 214]
    # for i in [12, 500, 10, 11, 38, 142, 186, 193, 214]:
    for i  in [12, 56]:
    # for i in range(dataset.__len__()):
        # if i in run_list:
            # continue
        G_ema = copy.deepcopy(G)  # deepcopy can make sure they are correct.
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        os.makedirs(os.path.join(run_dir, str(i).zfill(6)), exist_ok=True)
        image, label, mask, normal = dataset[i]
        # skimage.io.imsave(os.path.join(run_dir, "{}_proj_mask.png".format(str(i).zfill(3))), mask)
        c = torch.from_numpy(label).to(device).unsqueeze(0)
        target_uint8 = np.array(image, dtype=np.uint8).transpose([1, 2, 0])
        target_pil = PIL.Image.fromarray(target_uint8)
        normal_uint8 = np.array(normal, dtype=np.uint8).transpose([1, 2, 0])
        normal_pil = PIL.Image.fromarray(normal_uint8)
        # Optimize projection.
        start_time = perf_counter()

        projected_w_steps, projected_w_geo_steps, z_sample, z_geo_sample = project(
            G_ema,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
            target_normal=torch.tensor(normal_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
            c=c,
            num_steps=num_steps,
            device=device,
            use_normal_map=use_normal_map,
            verbose=True
        )

        print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
        if num_steps_pti > 0:
            G_steps = project_pti(
                G_ema,
                target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
                target_normal=torch.tensor(normal_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
                w_pivot=projected_w_steps[-1:],
                w_geo_pivot=projected_w_geo_steps[-1:],
                c=c,
                num_steps=num_steps_pti,
                device=device,
                use_normal_map=use_normal_map,
                verbose=True
            )
            print (f'Elapsed: {(perf_counter()-start_time):.1f} s')
        
        # Render debug output: optional video and projected image and W vector.
        G_final = None
        save_video = True
        if save_video:
            video = imageio.get_writer(f'{os.path.join(run_dir, str(i).zfill(6))}/proj.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
            print (f'Saving optimization progress video "{os.path.join(run_dir, str(i).zfill(6))}/proj.mp4"')
            for projected_w, projected_w_geo in zip(projected_w_steps[::4],projected_w_geo_steps[::4]):
                synth_image, _, synth_normal, _, _, _, _, _, _, _, _, _ = G_ema.generate_3d(
                    ws=projected_w.unsqueeze(0).to(device), 
                    ws_geo=projected_w_geo.unsqueeze(0).to(device), 
                    z=None, geo_z=None, c=c, noise_mode='const', only_img=False,
                    return_depth=False, generate_no_light=True, truncation_psi=0.7)
                synth_image, synth_mask = synth_image[:, :3], synth_image[:, 3:]
                synth_normal = synth_normal[:, :3]
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                synth_normal = (synth_normal + 1) * (255/2)
                synth_normal = synth_normal.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(np.concatenate([target_uint8, synth_image, synth_normal], axis=1))
                
            if num_steps_pti > 0:
                for G_new_state_dict in G_steps:
                    G_new = G_ema.eval().cpu()
                    G_new.load_state_dict(G_new_state_dict)
                    G_new.to(device)
                    synth_image, mask, synth_normal, _, _, _, _, _, _, _, _, _ = G_new.generate_3d(
                        ws=projected_w_steps[-1].unsqueeze(0).to(device), 
                        ws_geo=projected_w_geo_steps[-1].unsqueeze(0).to(device), 
                        z=None, geo_z=None, c=c, noise_mode='const', only_img=False,
                        return_depth=False, generate_no_light=True, truncation_psi=0.7)
                    synth_image, synth_mask = synth_image[:, :3], synth_image[:, 3:]
                    synth_normal = synth_normal[:, :3]
                    normal_mask = mask.permute(0, 3, 1, 2)
                    background = torch.ones_like(synth_normal)
                    synth_normal = synth_normal * normal_mask + background * (1 - normal_mask)
                    synth_image = (synth_image + 1) * (255/2)
                    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    synth_normal = (synth_normal + 1) * (255/2)
                    synth_normal = synth_normal.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    video.append_data(np.concatenate([target_uint8, synth_image, synth_normal], axis=1))
                    G_new.cpu()
                G_final = G_new.to(device)

            if G_final is None:
                G_final = G_ema

            # render 360 degree video
            if save_multiview:
                gen_multiview_video(
                    G_final, video=video, c=c,
                    ws=projected_w_steps[-1].unsqueeze(0).to(device), 
                    ws_geo=projected_w_geo_steps[-1].unsqueeze(0).to(device), 
                    target_image=target_uint8, run_dir=run_dir, idx=i)
        
            # animation
            if save_animation:
                gen_animation_video(
                    G_final, video=video, c=c,
                    ws=projected_w_steps[-1].unsqueeze(0).to(device), 
                    ws_geo=projected_w_geo_steps[-1].unsqueeze(0).to(device), 
                    target_image=target_uint8, run_dir=run_dir, idx=i
                )

            video.close()
        
        if G_final is None:
            G_final = G_ema

        # Save final projected frame and W vector
        target_pil.save(f'{os.path.join(run_dir, str(i).zfill(6))}/target.png')
        normal_pil.save(f'{os.path.join(run_dir, str(i).zfill(6))}/target_normal.png')
        projected_w = projected_w_steps[-1]
        projected_w_geo = projected_w_geo_steps[-1]
        synth_image, mask, synth_normal, _, _, _, _, _, _, _, _, _ = G_final.generate_3d(
            ws=projected_w.unsqueeze(0).to(device), 
            ws_geo=projected_w_geo.unsqueeze(0).to(device), 
            z=None, geo_z=None, c=c, noise_mode='const', only_img=False,
            return_depth=False, generate_no_light=True, truncation_psi=0.7)
        synth_image, synth_mask = synth_image[:, :3], synth_image[:, 3:]
        synth_normal = synth_normal[:, :3]
        normal_mask = mask.permute(0, 3, 1, 2)
        background = torch.ones_like(synth_normal)
        synth_normal = synth_normal * normal_mask + background * (1 - normal_mask)
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        synth_normal = (synth_normal + 1) * (255/2)
        synth_normal = synth_normal.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        synth_mask = synth_mask[0].permute(1,2,0).cpu().numpy()
        synth_mask[synth_mask<=0.5] = 0
        synth_mask[synth_mask>0.5] = 1
        synth_mask *= 255
        synth_mask = synth_mask.astype(np.uint8)
        PIL.Image.fromarray(synth_image, 'RGB').save(f'{os.path.join(run_dir, str(i).zfill(6))}/proj.png')
        PIL.Image.fromarray(synth_normal, 'RGB').save(f'{os.path.join(run_dir, str(i).zfill(6))}/proj_normal.png')
        # PIL.Image.fromarray(synth_mask, 'L').save(f'{os.path.join(run_dir, str(i).zfill(6))}/proj_mask.png')
        np.savez(f'{os.path.join(run_dir, str(i).zfill(6))}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())
        np.savez(f'{os.path.join(run_dir, str(i).zfill(6))}/projected_w_geo.npz', w=projected_w_geo.unsqueeze(0).cpu().numpy())
        np.savez(f'{os.path.join(run_dir, str(i).zfill(6))}/z_samples.npz', w=z_sample.detach().cpu().numpy())
        np.savez(f'{os.path.join(run_dir, str(i).zfill(6))}/z_geo_samples.npz', w=z_geo_sample.detach().cpu().numpy())


def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    target_normal: torch.Tensor,
    c: torch.Tensor,
    *,
    ws_geo                     = None,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    optimize_noise             = False,
    verbose                    = False,
    use_normal_map             = False,
    psi                        = 1.0,
    device: torch.device
):
    
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)
    
    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    
    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(G.num_ws, G.z_dim)
    z_samples = torch.from_numpy(z_samples).to(device)
    z_samples.requires_grad = True

    z_geo_samples = np.random.RandomState(123).randn(G.num_ws_geo, G.z_dim)
    z_geo_samples = torch.from_numpy(z_geo_samples).to(device)
    z_geo_samples.requires_grad = True

    z_samples_0 = z_samples.clone()
    z_geo_samples_0 = z_geo_samples.clone()

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_normals = target_normal.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_normals_perc = (target_normals + 1) * (255/2)
    if target_normals_perc.shape[2] > 256:
        target_normals_perc = F.interpolate(target_normals_perc, size=(256, 256), mode='area')
    target_normal_features = vgg16(target_normals_perc, resize_images=False, return_lpips=True)

    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    canonical_mapping_kwargs = G.synthesis.get_canonical_mapping_quick(c)
    target_face_images = G.synthesis.get_part(target_images, c, canonical_mapping_kwargs)
    target_face_images_perc = (target_face_images + 1) * (255/2)
    target_face_images_perc = F.interpolate(target_face_images_perc, size=(224, 224), mode='area')
    target_face_features = vgg16(target_face_images_perc, resize_images=False, return_lpips=True)

    w_out = torch.zeros([num_steps, G.num_ws, G.w_dim], dtype=torch.float32, device="cpu")
    w_out_geo = torch.zeros([num_steps, G.num_ws_geo, G.w_dim], dtype=torch.float32, device="cpu")

    # if optimize_noise:
    #     optimizer = torch.optim.Adam([z_samples] + [z_geo_samples] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    # else:
    #     optimizer = torch.optim.Adam([z_samples] + [z_geo_samples] , betas=(0.9, 0.999), lr=initial_learning_rate)

    if optimize_noise:
        optimizer = torch.optim.Adam([z_samples] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
    else:
        optimizer = torch.optim.Adam([z_samples] , betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    if optimize_noise:
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        c_cam_perm = torch.zeros_like(c[:, :G.c_cam_dim])
        ws = G.mapping(z_samples, c=c_cam_perm, truncation_psi=psi)[:, 0].unsqueeze(0)
        ws_geo = G.mapping_geo(z_geo_samples, c=c_cam_perm, truncation_psi=psi)[:, 0].unsqueeze(0)
        synth_images, _, synth_normals, _, _, _, _, _, _, _, _, _ = G.generate_3d(
            z=z_samples, geo_z=z_geo_samples,
            ws=ws, ws_geo=ws_geo, c=c, noise_mode='const', only_img=False,
            return_depth=False, generate_no_light=True, truncation_psi=psi)

        synth_images, synth_masks = synth_images[:, :3], synth_images[:, 3:]
        synth_normals = synth_normals[:, :3]

        synth_face_images = G.synthesis.get_part(synth_images, c, canonical_mapping_kwargs)
        
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        synth_normals_perc = (synth_normals + 1) * (255/2)
        if synth_normals_perc.shape[2] > 256:
            synth_normals_perc = F.interpolate(synth_normals_perc, size=(256, 256), mode='area')

        synth_face_images_perc = (synth_face_images + 1) * (255/2)
        synth_face_images_perc = F.interpolate(synth_face_images_perc, size=(224, 224), mode='area')
        
        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        synth_normal_features = vgg16(synth_normals_perc, resize_images=False, return_lpips=True)
        synth_face_features = vgg16(synth_face_images_perc, resize_images=False, return_lpips=True)
        
        # Calc loss
        perc_loss = (target_features - synth_features).square().sum(1).mean()
        perc_face_loss = (target_face_features - synth_face_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()
        mse_face_loss = (target_face_images - synth_face_images).square().mean()

        if use_normal_map:
            normal_loss = (target_normals - synth_normals).square().mean()
            normal_perc_loss = (target_normal_features - synth_normal_features).square().mean()
        # w_norm_loss = (z_samples - z_samples_0).square().mean()*0.5 + (z_geo_samples - z_geo_samples_0).square().mean()*0.5
        # std_loss = ws.std(dim=1).mean()*0.5 + ws_geo.std(dim=1).mean()*0.5

        w_norm_loss = (z_samples - z_samples_0).square().mean()
        std_loss = ws.std(dim=1).mean()

        # Noise regularization.
        reg_loss = 0.0
        if optimize_noise:
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)

        loss = 0*std_loss + mse_loss*1 + mse_face_loss*0.0 + perc_loss*1 + perc_face_loss*0.0 + 0.01 * w_norm_loss + reg_loss * regularize_noise_weight
        if use_normal_map:
            loss = loss + normal_loss + normal_perc_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if use_normal_map:
            logprint(f'step: {step+1:>4d}/{num_steps} std: {std_loss:<4.2f} mse: {mse_loss:<4.2f} m_face: {mse_face_loss:<4.2f} normal: {normal_loss:<4.2f} perc: {perc_loss:<4.2f} p_face: {perc_face_loss:<4.2f} w_norm: {w_norm_loss:<4.2f} noise: {float(reg_loss):<5.2f}')
        else:
            logprint(f'step: {step+1:>4d}/{num_steps} std: {std_loss:<4.2f} mse: {mse_loss:<4.2f} m_face: {mse_face_loss:<4.2f} perc: {perc_loss:<4.2f} p_face: {perc_face_loss:<4.2f} w_norm: {w_norm_loss:<4.2f} noise: {float(reg_loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = ws.detach().cpu()[0]
        w_out_geo[step] = ws_geo.detach()[0]
        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    if w_out.shape[1] == 1:
        w_out = w_out.repeat([1, G.backbone.num_ws, 1])
    if w_out_geo.shape[1] == 1:
        w_out_geo = w_out_geo.repeat([1, G.backbone.num_ws, 1])
    return w_out, w_out_geo, z_samples, z_geo_samples


def project_pti(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    target_normal: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    w_pivot: torch.Tensor,
    w_geo_pivot: torch.Tensor,
    c: torch.Tensor,
    *,
    num_steps                  = 1000,
    initial_learning_rate      = 3e-4,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    use_normal_map             = False,
    verbose                    = False,
    psi                        = 1.0,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).train().requires_grad_(True).to(device) # type: ignore

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_normals = target_normal.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_normals_perc = (target_normals + 1) * (255/2)
    if target_normals_perc.shape[2] > 256:
        target_normals_perc = F.interpolate(target_normals_perc, size=(256, 256), mode='area')
    target_normal_features = vgg16(target_normals_perc, resize_images=False, return_lpips=True)
    
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 255.0 * 2 - 1
    target_images_perc = (target_images + 1) * (255/2)
    if target_images_perc.shape[2] > 256:
        target_images_perc = F.interpolate(target_images_perc, size=(256, 256), mode='area')
    target_features = vgg16(target_images_perc, resize_images=False, return_lpips=True)

    canonical_mapping_kwargs = G.synthesis.get_canonical_mapping_quick(c)
    target_face_images = G.synthesis.get_part(target_images, c, canonical_mapping_kwargs)
    target_face_images_perc = (target_face_images + 1) * (255/2)
    target_face_images_perc = F.interpolate(target_face_images_perc, size=(224, 224), mode='area')
    target_face_features = vgg16(target_face_images_perc, resize_images=False, return_lpips=True)

    w_pivot = w_pivot.to(device).detach()
    w_geo_pivot = w_geo_pivot.to(device).detach()
    optimizer = torch.optim.Adam(G.synthesis.parameters(), betas=(0.9, 0.999), lr=initial_learning_rate)

    out_params = []
    for step in range(num_steps):
        # gen_output = G.synthesis(w_pivot, w_geo_pivot, c=c, noise_mode='const')
        synth_images, _, synth_normals, _, _, _, _, _, _, _, _, _ = G.generate_3d(
            z=None, geo_z=None, ws=w_pivot, ws_geo=w_geo_pivot, c=c, 
            noise_mode='const', only_img=False, return_depth=False, 
            generate_no_light=True, truncation_psi=psi)
        
        synth_images, synth_masks = synth_images[:, :3], synth_images[:, 3:]
        synth_normals = synth_normals[:, :3]

        synth_face_images = G.synthesis.get_part(synth_images, c, canonical_mapping_kwargs)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images_perc = (synth_images + 1) * (255/2)
        if synth_images_perc.shape[2] > 256:
            synth_images_perc = F.interpolate(synth_images_perc, size=(256, 256), mode='area')

        synth_normals_perc = (synth_normals + 1) * (255/2)
        if synth_normals_perc.shape[2] > 256:
            synth_normals_perc = F.interpolate(synth_normals_perc, size=(256, 256), mode='area')

        synth_face_images_perc = (synth_face_images + 1) * (255/2)
        synth_face_images_perc = F.interpolate(synth_face_images_perc, size=(224, 224), mode='area')
        
        # Features for synth images.
        synth_features = vgg16(synth_images_perc, resize_images=False, return_lpips=True)
        synth_normal_features = vgg16(synth_normals_perc, resize_images=False, return_lpips=True)
        synth_face_features = vgg16(synth_face_images_perc, resize_images=False, return_lpips=True)

        # Calc loss
        perc_loss = (target_features - synth_features).square().sum(1).mean()
        perc_face_loss = (target_face_features - synth_face_features).square().sum(1).mean()

        mse_loss = (target_images - synth_images).square().mean()
        mse_face_loss = (target_face_images - synth_face_images).square().mean()

        if use_normal_map:
            normal_loss = (target_normals - synth_normals).square().mean()
            normal_perc_loss = (target_normal_features - synth_normal_features).square().mean()
        loss = mse_loss*0.1 + perc_loss*1 + mse_face_loss*1.0 + perc_face_loss*1
        if use_normal_map:
            loss = loss + normal_loss*0.1 + normal_perc_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if use_normal_map:
            logprint(f'step: {step+1:>4d}/{num_steps} mse: {mse_loss:<4.2f} m_face: {mse_face_loss:<4.2f} normal: {normal_loss:<4.2f} perc: {perc_loss:<4.2f} p_face: {perc_face_loss:<4.2f}')
        else:
            logprint(f'step: {step+1:>4d}/{num_steps} mse: {mse_loss:<4.2f} m_face: {mse_face_loss:<4.2f} perc: {perc_loss:<4.2f} p_face: {perc_face_loss:<4.2f}')

        if step == num_steps - 1 or step % 10 == 0:
            # out_params.append(copy.deepcopy(G).eval().requires_grad_(False).cpu())
            new_state_dict = copy.deepcopy(G.state_dict())
            for k, v in new_state_dict.items():
                new_state_dict[k] = copy.deepcopy(v).cpu()
            out_params.append(new_state_dict)

    return out_params


def load_pickle_file(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def load_mixamo_smpl(actions_dir, action_type='0007', skip=1):
    result = load_pickle_file(os.path.join(actions_dir, action_type, 'result.pkl'))

    anim_len = result['anim_len']
    pose_array = result['smpl_array'].reshape(anim_len, -1)
    cam_array = result['cam_array']
    mocap = []
    for i in range(0, anim_len, skip):
        mocap.append({
            'cam': cam_array[i],
            'global_orient': pose_array[i, :3],
            'body_pose': pose_array[i, 3:72],
            'transl': np.array([cam_array[i, 1], cam_array[i, 2], 0])
            })

    return mocap


def gen_multiview_video(G, c, ws, ws_geo, video, target_image, run_dir, idx):
    os.makedirs(os.path.join(run_dir, str(idx).zfill(6), 'multiview'), exist_ok=True)
    assert G.c_cam_dim == 16
    c_cam = c[:, :G.c_cam_dim].clone()
    c_smpl = c[:, G.c_cam_dim:].clone()
    rots = [i for i in range(0,180,5)]+[i for i in range(180, -270, -5)]
    for t in tqdm(range(120)):
        t = t / 120
        azim = math.pi * 1 * np.cos(t * 1 * math.pi)
        elev = math.pi / 2

        cam2world_matrix = LookAtPoseSampler.sample(azim, elev, torch.tensor([0, -0.3, 0], device=c.device), radius=2.3, device=c.device)
        world2cam_matrix = torch.linalg.inv(cam2world_matrix)
        # c_cam_rot = c_cam.clone().reshape(1, 4, 4)
        # c_cam_rot[:, :3, :3] = torch.bmm(world2cam_matrix[:, :3, :3], c_cam_rot[:, :3, :3])#.reshape(1, -1)
        # c_cam_rot = c_cam_rot.reshape(1, -1)
        c_cam_rot = world2cam_matrix.reshape(1, -1)
        c_input = torch.cat((c_cam_rot, c_smpl.reshape(1, -1)), dim=-1).reshape(1, -1)
        synth_image, mask, synth_normal, _, _, _, _, _, _, _, _, _ = G.generate_3d(
            ws=ws, ws_geo=ws_geo, z=None, geo_z=None, c=c_input, 
            noise_mode='const', only_img=False, return_depth=False, 
            generate_no_light=True, truncation_psi=0.7)
        synth_image, synth_mask = synth_image[:, :3], synth_image[:, 3:]
        synth_normal = synth_normal[:, :3]
        normal_mask = mask.permute(0, 3, 1, 2)
        background = torch.ones_like(synth_normal)
        synth_normal = synth_normal * normal_mask + background * (1 - normal_mask)
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        synth_normal = (synth_normal + 1) * (255/2)
        synth_normal = synth_normal.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        synth_mask = synth_mask[0].permute(1,2,0).cpu().numpy()
        synth_mask[synth_mask<=0.5] = 0
        synth_mask[synth_mask>0.5] = 1
        synth_mask *= 255
        synth_mask = synth_mask.astype(np.uint8)
        video.append_data(np.concatenate([target_image, synth_image, synth_normal], axis=1))
        t = int(t * 120)
        skimage.io.imsave(os.path.join(run_dir, str(idx).zfill(6), 'multiview', '{}.png'.format(str(t).zfill(3))), np.concatenate([target_image, synth_image, synth_normal]))
        # skimage.io.imsave(os.path.join(run_dir, str(idx).zfill(6), 'multiview', '{}_mask.png'.format(str(t).zfill(3))), synth_mask)


def gen_animation_video(G, c, ws, ws_geo, video, target_image, action_dir="smplx/mocap/mixamo", action_type="0145", frame_skip=1, run_dir=None, idx=None):
    os.makedirs(os.path.join(run_dir, str(idx).zfill(6), 'animation'), exist_ok=True)
    assert G.c_cam_dim == 16
    mocap = load_mixamo_smpl(action_dir, action_type, frame_skip)
    c_cam = c[:, :G.c_cam_dim].clone()
    c_smpl = c[:, G.c_cam_dim:].clone()
    mocap_len = len(mocap[:200])
    for t in tqdm(range(mocap_len)):
        mocap_data = mocap[t]
        mocap_params = {
            'body_pose': torch.from_numpy(mocap_data['body_pose']).reshape(1, -1).to(c.device).float(), 
            'global_orient': torch.from_numpy(mocap_data['global_orient']).reshape(1, -1).to(c.device).float(),
            'transl': torch.from_numpy(mocap_data['transl']).reshape(1, -1).to(c.device).float()
        }
        c_smpl_input = c_smpl.clone()
        c_smpl_input[:, 3:72] = mocap_params['body_pose']
        c_smpl_input[:, 0:3] = mocap_params['global_orient']
        c_input = torch.cat((c_cam, c_smpl_input.reshape(1, -1)), dim=-1).reshape(1, -1)
        synth_image, mask, synth_normal, _, _, _, _, _, _, _, _, _ = G.generate_3d(
            ws=ws, ws_geo=ws_geo, z=None, geo_z=None, c=c_input, 
            noise_mode='const', only_img=False, return_depth=False, 
            generate_no_light=True, truncation_psi=0.7)
        synth_image, synth_mask = synth_image[:, :3], synth_image[:, 3:]
        synth_normal = synth_normal[:, :3]
        normal_mask = mask.permute(0, 3, 1, 2)
        background = torch.ones_like(synth_normal)
        synth_normal = synth_normal * normal_mask + background * (1 - normal_mask)
        if True:
            synth_image = torch.flip(synth_image, dims=[2,3])
            synth_normal = torch.flip(synth_normal, dims=[2,3])
            synth_mask = torch.flip(synth_mask, dims=[2,3])
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        synth_normal = (synth_normal + 1) * (255/2)
        synth_normal = synth_normal.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        synth_mask = synth_mask[0].permute(1,2,0).cpu().numpy()
        synth_mask[synth_mask<=0.5] = 0
        synth_mask[synth_mask>0.5] = 1
        synth_mask *= 255
        synth_mask = synth_mask.astype(np.uint8)
        video.append_data(np.concatenate([target_image, synth_image, synth_normal], axis=1))
        skimage.io.imsave(os.path.join(run_dir, str(idx).zfill(6), 'animation', '{}.png'.format(str(int(t)).zfill(3))), np.concatenate([target_image, synth_image, synth_normal]))
        # skimage.io.imsave(os.path.join(run_dir, str(idx).zfill(6), 'animation', '{}_mask.png'.format(str(int(t)).zfill(3))), synth_mask)
