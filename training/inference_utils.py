# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
'''
Utily functions for the inference
'''
import torch
import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import PIL.Image
from training.utils.utils_3d import save_obj, savemeshtes2
import imageio
import cv2
from tqdm import tqdm
from training.utils.renderer_utils import create_cameras, create_mesh_normal_renderer
from pytorch3d.structures import Meshes
import pyrender
import trimesh
from training.utils.pyrender_warpper import render_normal_and_depth_buffers


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    gw = _N // gh
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if not fname is None:
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)
    return img


def save_3d_shape(mesh_v_list, mesh_f_list, root, idx):
    n_mesh = len(mesh_f_list)
    mesh_dir = os.path.join(root, 'mesh_pred')
    os.makedirs(mesh_dir, exist_ok=True)
    for i_mesh in range(n_mesh):
        mesh_v = mesh_v_list[i_mesh]
        mesh_f = mesh_f_list[i_mesh]
        mesh_name = os.path.join(mesh_dir, '%07d_%02d.obj' % (idx, i_mesh))
        save_obj(mesh_v, mesh_f, mesh_name)


def gen_swap(ws_geo_list, ws_tex_list, camera, generator, save_path, gen_mesh=False, ):
    '''
    With two list of latent code, generate a matrix of results, N_geo x N_tex
    :param ws_geo_list: the list of geometry latent code
    :param ws_tex_list: the list of texture latent code
    :param camera:  camera to render the generated mesh
    :param generator: GET3D_Generator
    :param save_path: path to save results
    :param gen_mesh: whether we generate textured mesh
    :return:
    '''
    img_list = []
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for i_geo, ws_geo in enumerate(ws_geo_list):
            for i_tex, ws_tex in enumerate(ws_tex_list):
                img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
                sdf_reg_loss, render_return_value = generator.synthesis.generate(
                    ws_tex.unsqueeze(dim=0), update_emas=None, camera=camera,
                    update_geo=None, ws_geo=ws_geo.unsqueeze(dim=0),
                )
                img_list.append(img[:, :3].data.cpu().numpy())
                if gen_mesh:
                    generated_mesh = generator.synthesis.extract_3d_shape(ws_tex.unsqueeze(dim=0), ws_geo.unsqueeze(dim=0))
                    for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
                        savemeshtes2(
                            mesh_v.data.cpu().numpy(),
                            all_uvs.data.cpu().numpy(),
                            mesh_f.data.cpu().numpy(),
                            all_mesh_tex_idx.data.cpu().numpy(),
                            os.path.join(save_path, '%02d_%02d.obj' % (i_geo, i_tex))
                        )
                        lo, hi = (-1, 1)
                        img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
                        img = (img - lo) * (255 / (hi - lo))
                        img = img.clip(0, 255)
                        mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
                        mask = (mask <= 3.0).astype(np.float)
                        kernel = np.ones((3, 3), 'uint8')
                        dilate_img = cv2.dilate(img, kernel, iterations=1)
                        img = img * (1 - mask) + dilate_img * mask
                        img = img.clip(0, 255).astype(np.uint8)
                        PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                            os.path.join(save_path, '%02d_%02d.png' % (i_geo, i_tex)))
    img_list = np.concatenate(img_list, axis=0)
    img = save_image_grid(img_list, os.path.join(save_path, 'inter_img.jpg'), drange=[-1, 1], grid_size=[ws_tex_list.shape[0], ws_geo_list.shape[0]])
    return img


def save_visualization_for_interpolation(
        generator, num_sam=10, c_to_compute_w_avg=None, save_dir=None, gen_mesh=False):
    '''
    Interpolate between two latent code and generate a swap between them
    :param generator: GET3D generator
    :param num_sam: number of samples we hope to generate
    :param c_to_compute_w_avg: None is default
    :param save_dir: path to save
    :param gen_mesh: whether we want to generate 3D textured mesh
    :return:
    '''
    with torch.no_grad():
        generator.update_w_avg(c_to_compute_w_avg)
        geo_codes = torch.randn(num_sam, generator.z_dim, device=generator.device)
        tex_codes = torch.randn(num_sam, generator.z_dim, device=generator.device)
        ws_geo = generator.mapping_geo(geo_codes, None, truncation_psi=0.7)
        ws_tex = generator.mapping(tex_codes, None, truncation_psi=0.7)
        camera_list = [generator.synthesis.generate_rotate_camera_list(n_batch=num_sam)[4]]

        select_geo_codes = np.arange(4)  # You can change to other selected shapes
        select_tex_codes = np.arange(4)
        for i in range(len(select_geo_codes) - 1):
            ws_geo_a = ws_geo[select_geo_codes[i]].unsqueeze(dim=0)
            ws_geo_b = ws_geo[select_geo_codes[i + 1]].unsqueeze(dim=0)
            ws_tex_a = ws_tex[select_tex_codes[i]].unsqueeze(dim=0)
            ws_tex_b = ws_tex[select_tex_codes[i + 1]].unsqueeze(dim=0)
            new_ws_geo = []
            new_ws_tex = []
            n_interpolate = 10
            for _i in range(n_interpolate):
                w = float(_i + 1) / n_interpolate
                w = 1 - w
                new_ws_geo.append(ws_geo_a * w + ws_geo_b * (1 - w))
                new_ws_tex.append(ws_tex_a * w + ws_tex_b * (1 - w))
            new_ws_tex = torch.cat(new_ws_tex, dim=0)
            new_ws_geo = torch.cat(new_ws_geo, dim=0)
            save_path = os.path.join(save_dir, 'interpolate_%02d' % (i))
            os.makedirs(save_path, exist_ok=True)
            gen_swap(
                new_ws_geo, new_ws_tex, camera_list[0], generator,
                save_path=save_path, gen_mesh=gen_mesh
            )


def save_visualization(
        G_ema, grid_z, grid_c, run_dir, cur_nimg, grid_size, cur_tick,
        image_snapshot_ticks=50,
        save_gif_name=None,
        save_all=True,
        grid_tex_z=None,
):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''
    # with torch.no_grad():
    #     G_ema.update_w_avg()
        # camera_list = G_ema.synthesis.generate_rotate_camera_list(n_batch=grid_z[0].shape[0])
        # camera_img_list = []
        # if not save_all:
        #     camera_list = [camera_list[4]]  # we only save one camera for this
    if True:
        G_ema.update_w_avg()

        if grid_tex_z is None:
            grid_tex_z = grid_z
        # for i_camera, camera in enumerate(camera_list):
        images_list = []
        images_part_list = []
        mesh_v_list = []
        mesh_f_list = []
        for z, geo_z, c in zip(grid_tex_z, grid_z, grid_c):
            G_ema.requires_grad_(True)
            img, mask, normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, tex_hard_mask, part_img, normal_part_img = G_ema.generate_3d(
                z=z, geo_z=geo_z, c=c, noise_mode='const',
                generate_no_light=True, truncation_psi=0.7)
            G_ema.requires_grad_(False)
            rgb_img = img[:, :3]
            normal_map = normal_img[:, :3]
            save_img = torch.cat([rgb_img, mask.permute(0, 3, 1, 2).expand(-1, 3, -1, -1), normal_map], dim=-1).detach()
            images_list.append(save_img.cpu().numpy())
            mesh_v_list.extend([v.data.cpu().numpy() for v in mesh_v])
            mesh_f_list.extend([f.data.cpu().numpy() for f in mesh_f])

            if G_ema.synthesis.part_disc:
                rgb_img_part = part_img[:, :3]
                normal_map_part = normal_part_img[:, :3]
                mask_part = part_img[:, -1:].expand(-1, 3, -1, -1)
                save_img_part = torch.cat([rgb_img_part, mask_part, normal_map_part], dim=-1).detach()
                images_part_list.append(save_img_part.cpu().numpy())

        images = np.concatenate(images_list, axis=0)
        if save_gif_name is None:
            save_file_name = 'fakes'
        else:
            save_file_name = 'fakes_%s' % (save_gif_name.split('.')[0])
        if G_ema.synthesis.part_disc:
            images_part = np.concatenate(images_part_list, axis=0)
            if save_gif_name is None:
                save_file_name_part = 'fakes'
            else:
                save_file_name_part = 'fakes_%s' % (save_gif_name.split('.')[0])
            img_part = save_image_grid(
                images_part, os.path.join(
                    run_dir,
                    f'{save_file_name_part}_{cur_nimg // 1000:06d}_part.png'),
                drange=[-1, 1], grid_size=grid_size)
        img = save_image_grid(
            images, os.path.join(
                run_dir,
                f'{save_file_name}_{cur_nimg // 1000:06d}.png'),
            drange=[-1, 1], grid_size=grid_size)
        # camera_img_list.append(img)
        if save_gif_name is None:
            save_gif_name = f'fakes_{cur_nimg // 1000:06d}.gif'
        # if save_all:
        #     imageio.mimsave(os.path.join(run_dir, save_gif_name), camera_img_list)
        n_shape = 10  # we only save 10 shapes to check performance
        if cur_tick % min((image_snapshot_ticks * 20), 100) == 0:
            save_3d_shape(mesh_v_list[:n_shape], mesh_f_list[:n_shape], run_dir, cur_nimg // 100)


def save_visualization_inference_for_depth_eval(
    gen_iter, G_ema, batch_gen=None, max_items=0, 
    device='cuda', truncation_psi=1.0, outdir=None
):
    # Initialize.
    teval_imgs = 0
    # Main loop.
    seed = 0

    with tqdm(total=max_items) as pbar:
        while teval_imgs < max_items:
            # GET GT AND GENERATED RESULT IN THE SAME CAMERA POSE AND SMPL POSE
            _, gt_label, _, _ = next(gen_iter)
            gt_label = gt_label.to(device)
            # Generate fake image based on gt_label
            z = torch.from_numpy(np.random.RandomState(seed).randn(batch_gen, G_ema.z_dim)).to(device)
            geo_z = torch.from_numpy(np.random.RandomState(seed).randn(batch_gen, G_ema.z_dim)).to(device)
            pred_img, _, _, _, _, _, _, _, _, _, _, _, pred_depth = G_ema.generate_3d(
                z=z, geo_z=geo_z, c=gt_label, noise_mode='const', only_img=True,
                return_depth=True, generate_no_light=True, truncation_psi=0.7)
            pred_img = pred_img[:, :3]
            pred_img = (pred_img * 127.5+128).clamp(0, 255).to(torch.uint8).permute(0,2,3,1).cpu().numpy()
            pred_depth = -pred_depth.permute(0,2,3,1).cpu().numpy()
            for i in range(pred_img.shape[0]):
                img = pred_img[i]
                depth = pred_depth[i]
                depth = ((depth-depth.min()) / (depth.max()-depth.min())) * 255
                np.save(f"{outdir}/depth/img_{seed}.npy", img)
                np.save(f"{outdir}/depth/depth_{seed}.npy", depth)

            teval_imgs += batch_gen
            pbar.update(batch_gen)
            seed += 1

def save_visualization_inference_for_pck_eval(
    gen_iter, G_ema, batch_size=64, batch_gen=None, 
    pose_model=None, det_model=None, max_items=0, 
    device='cuda', truncation_psi=1.0, outdir=None
):
    from mmpose.core.evaluation.top_down_eval import keypoint_pck_accuracy

    # Initialize.
    teval_imgs = 0
    # Main loop.
    hits = 0
    total = 0
    seed = 0

    with tqdm(total=max_items) as pbar:
        while teval_imgs < max_items:
            # GET GT AND GENERATED RESULT IN THE SAME CAMERA POSE AND SMPL POSE
            gt_img, gt_label, _, _ = next(gen_iter)
            gt_label = gt_label.to(device)
            # Generate fake image based on gt_label
            z = torch.from_numpy(np.random.RandomState(seed).randn(batch_gen, G_ema.z_dim)).to(device)
            geo_z = torch.from_numpy(np.random.RandomState(seed).randn(batch_gen, G_ema.z_dim)).to(device)
            pred_img, _, _, _, _, _, _, _, _, _, _, _ = G_ema.generate_3d(
                z=z, geo_z=geo_z, c=gt_label, noise_mode='const', only_img=True,
                generate_no_light=True, truncation_psi=0.7)
            pred_img = pred_img[:, :3]
            pred_img = (pred_img * 127.5+128).clamp(0, 255).to(torch.uint8)
            vis = False if teval_imgs % 500 != 0 else True
            save_idx = batch_gen * seed
            gt_kpts, gt_failed = compute_kpts(gt_img, pose_model, det_model, 'gt', vis=vis, outdir=outdir, save_idx=save_idx)
            pred_kpts, pred_failed = compute_kpts(pred_img, pose_model, det_model, 'pred', vis=vis, outdir=outdir, save_idx=save_idx)
            gt_scores = gt_kpts[..., -1]
            gt_kpts = gt_kpts[..., :2]
            pred_scores = pred_kpts[..., -1]
            pred_kpts = pred_kpts[..., :2]

            det_thres = 0.8
            mask = np.logical_and((gt_scores > det_thres), (pred_scores > det_thres))
            mask = np.logical_and(mask, gt_scores[:, 8, None] > det_thres)
            mask = np.logical_and(mask, gt_scores[:, 9, None] > det_thres)
            thr = 0.5
            interocular = np.linalg.norm(gt_kpts[:, 8, :] - gt_kpts[:, 9, :], axis=1, keepdims=True)
            normalize = np.tile(interocular, [1, 2])

            oe = keypoint_pck_accuracy(pred_kpts, gt_kpts, mask, thr, normalize)
            hits += oe[1] * oe[2] * pred_kpts.shape[0]
            total += oe[2] * pred_kpts.shape[0]

            teval_imgs += batch_gen
            pbar.update(batch_gen)
            seed += 1
    
    print('Total: {}'.format(total))
    return float(hits) / float(total)    

def save_visualization_inference(
        G_ema, grid_z, grid_c, run_dir, cur_nimg, grid_size, cur_tick,
        image_snapshot_ticks=50,
        save_gif_name=None,
        save_all=True,
        grid_tex_z=None,
):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''
    # with torch.no_grad():
    #     G_ema.update_w_avg()
        # camera_list = G_ema.synthesis.generate_rotate_camera_list(n_batch=grid_z[0].shape[0])
        # camera_img_list = []
        # if not save_all:
        #     camera_list = [camera_list[4]]  # we only save one camera for this
    if True:
        G_ema.update_w_avg()
        if grid_tex_z is None:
            grid_tex_z = grid_z
        # for i_camera, camera in enumerate(camera_list):
        images_list = []
        mesh_v_list = []
        mesh_f_list = []
        for z, geo_z, c in zip(grid_tex_z, grid_z, grid_c):
            G_ema.requires_grad_(True)
            img, mask, normals_fine, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, tex_hard_mask, _, _ = G_ema.generate_3d(
                z=z, geo_z=geo_z, c=c, noise_mode='const',
                generate_no_light=True, truncation_psi=0.7)
                # camera=camera)
            G_ema.requires_grad_(False)

            use_pytorch3d = True
            if use_pytorch3d:
                world2cam_matrix = c[:,:16].view(4, 4)
                # cam2world_matrix = torch.linalg.inv(world2cam_matrix)
                aligned_matrix = torch.Tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
                ).float().to(world2cam_matrix.device)
                world2cam_matrix = aligned_matrix @ world2cam_matrix
                cam2world_matrix = torch.linalg.inv(world2cam_matrix)

                render_R = cam2world_matrix[:3, :3]
                render_T = torch.tensor([world2cam_matrix[0, 3], world2cam_matrix[1, 3], world2cam_matrix[2, 3]]).type_as(cam2world_matrix)

                fovy = np.arctan(32 / 2 / 35) * 2
                fovyangle = fovy / np.pi * 180.0
                cameras = create_cameras(R=render_R[None], T=render_T[None], fov=fovyangle, device=z.device)

                img_res = int(512)
                renderer = create_mesh_normal_renderer(
                            cameras, image_size=img_res,
                            light_location=((1.0,1.0,1.0),), specular_color=((0.2,0.2,0.2),),
                            ambient_color=((0.2,0.2,0.2),), diffuse_color=((0.65,.65,.65),),
                            device=z.device)
                sdf_mc_mesh = Meshes(
                    verts=mesh_v, 
                    faces=mesh_f,
                    textures=None
                )
                normals_coarse = renderer(sdf_mc_mesh)[..., :3].permute(0, 3, 1, 2) * (-1)

            rgb_img = img[:, :3]
            # normal_map = normal_img[:, :3]
            normals_fine = normals_fine[:, :3]

            cos_dis = (normals_coarse * normals_fine).sum(1, keepdims=True)

            sigma = 0.2
            fine_confidence = 0.5 * (cos_dis + 1) # 0~1
            fine_confidence = torch.exp(-(fine_confidence - 1)**2/2.0/sigma/sigma)
            fused_n = normals_fine * fine_confidence + normals_coarse * (1 - fine_confidence)
            # print('fine_confidence', torch.amax(fine_confidence), ' ', torch.amin(fine_confidence))
            normals_x = fused_n / torch.linalg.norm(fused_n, dim=1, keepdim=True)
            background = torch.zeros_like(normals_x)

            # tex_hard_mask = tex_hard_mask.permute(0, 3, 1, 2)
            # normal_map = tex_hard_mask * normals_x + (1 - tex_hard_mask) * background
            normal_mask = mask.permute(0, 3, 1, 2).expand(-1, 3, -1, -1)
            normal_map = normal_mask * normals_x + (1 - normal_mask) * background

            save_img = torch.cat([rgb_img, mask.permute(0, 3, 1, 2).expand(-1, 3, -1, -1), normals_fine, normals_coarse, normal_map], dim=-1).detach()
            images_list.append(save_img.cpu().numpy())
            mesh_v_list.extend([v.data.cpu().numpy() for v in mesh_v])
            mesh_f_list.extend([f.data.cpu().numpy() for f in mesh_f])

        images = np.concatenate(images_list, axis=0)
        if save_gif_name is None:
            save_file_name = 'fakes'
        else:
            save_file_name = 'fakes_%s' % (save_gif_name.split('.')[0])
        # if save_all:
        #     img = save_image_grid(
        #         images, None,
        #         drange=[-1, 1], grid_size=grid_size)
        # else:
        img = save_image_grid(
            images, os.path.join(
                run_dir,
                f'{save_file_name}_{cur_nimg // 1000:06d}.png'),
            drange=[-1, 1], grid_size=grid_size)
        # camera_img_list.append(img)
        if save_gif_name is None:
            save_gif_name = f'fakes_{cur_nimg // 1000:06d}.gif'
        # if save_all:
        #     imageio.mimsave(os.path.join(run_dir, save_gif_name), camera_img_list)
        n_shape = 10  # we only save 10 shapes to check performance
        if cur_tick % min((image_snapshot_ticks * 20), 100) == 0:
            save_3d_shape(mesh_v_list[:n_shape], mesh_f_list[:n_shape], run_dir, cur_nimg // 100)

def save_visualization_full(
        G_ema, grid_z, grid_c, run_dir, cur_nimg, grid_size, cur_tick,
        image_snapshot_ticks=50,
        save_gif_name=None,
        save_all=True,
        grid_tex_z=None
):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''
    # with torch.no_grad():
    if True:
        G_ema.update_w_avg()
        # camera_list = G_ema.synthesis.generate_rotate_camera_list(n_batch=grid_z[0].shape[0])
        # camera_img_list = []
        # if not save_all:
        #     camera_list = [camera_list[4]]  # we only save one camera for this
        if grid_tex_z is None:
            grid_tex_z = grid_z
        # for i_camera, camera in enumerate(camera_list):
        images_list = []
        mesh_v_list = []
        mesh_f_list = []
        for z, geo_z, c in zip(grid_tex_z, grid_z, grid_c):
            G_ema.requires_grad_(True)
            img, mask, normal_img, sdf, deformation, v_deformed, mesh_v, mesh_f, img_wo_light, tex_hard_mask, _, _ = G_ema.generate_3d(
                z=z, geo_z=geo_z, c=c, noise_mode='const',
                generate_no_light=True, truncation_psi=0.7)
                # camera=camera)
            G_ema.requires_grad_(False)
            rgb_img = img[:, :3]
            normal_map = normal_img[:, :3]

            world2cam_matrix = c[:,:16].view(4, 4)
            cam2world_matrix = torch.linalg.inv(world2cam_matrix)

            dense_mesh_v, dense_mesh_f = G_ema.generate_dense_mesh(
                z=z, geo_z=geo_z, c=c, noise_mode='const',
                generate_raw=True, with_texture=False,truncation_psi=0.7)

            use_pytorch3d = False

            if use_pytorch3d:
                aligned_matrix = torch.Tensor(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
                ).float().to(world2cam_matrix.device)
                world2cam_matrix = aligned_matrix @ world2cam_matrix
                cam2world_matrix = torch.linalg.inv(world2cam_matrix)

                render_R = cam2world_matrix[:3, :3]
                render_T = torch.tensor([world2cam_matrix[0, 3], world2cam_matrix[1, 3], world2cam_matrix[2, 3]]).type_as(cam2world_matrix)

                fovy = np.arctan(32 / 2 / 35) * 2
                fovyangle = fovy / np.pi * 180.0
                cameras = create_cameras(R=render_R[None], T=render_T[None], fov=fovyangle, device=z.device)

                img_res = rgb_img.shape[-1]
                renderer = create_mesh_normal_renderer(
                            cameras, image_size=img_res,
                            light_location=((1.0,1.0,1.0),), specular_color=((0.2,0.2,0.2),),
                            ambient_color=((0.2,0.2,0.2),), diffuse_color=((0.65,.65,.65),),
                            device=z.device)
                sdf_mc_mesh = Meshes(
                    verts=dense_mesh_v,
                    faces=dense_mesh_f,
                    textures=None
                )
                normal_img_dense = renderer(sdf_mc_mesh)[..., :3].permute(0, 3, 1, 2)
            else:
                img_res = rgb_img.shape[-1]
                fov = 0.8575560450553894 # radians
                fx = fy = 0.5 / np.tan(0.5 * fov)
                fx = fx * img_res
                fy = fy * img_res

                camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy,
                                                    cx=img_res/2, cy=img_res/2)
                
                camera_transform = cam2world_matrix.detach().cpu().numpy()

                mesh = trimesh.Trimesh(dense_mesh_v[0].data.cpu().numpy(), dense_mesh_f[0].data.cpu().numpy())

                render_normal_img, render_depth_img = render_normal_and_depth_buffers(mesh, camera, camera_transform, img_res)
                normal_img_dense = torch.tensor(render_normal_img.copy()).type_as(rgb_img).unsqueeze(0).permute(0, 3, 1, 2)  / 127.5 - 1
                #depth_img = torch.tensor(render_depth_img.copy()).type_as(rgb_img).unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

            save_img = torch.cat([rgb_img, mask.permute(0, 3, 1, 2).expand(-1, 3, -1, -1), normal_map, normal_img_dense], dim=-1).detach()

            images_list.append(save_img.cpu().numpy())
            mesh_v_list.extend([v.data.cpu().numpy() for v in dense_mesh_v])
            mesh_f_list.extend([f.data.cpu().numpy() for f in dense_mesh_f])

        images = np.concatenate(images_list, axis=0)

        if save_gif_name is None:
            save_file_name = 'fakes'
        else:
            save_file_name = 'fakes_%s' % (save_gif_name.split('.')[0])

        img = save_image_grid(
            images, os.path.join(
                run_dir,
                f'{save_file_name}_{cur_nimg // 1000:06d}.png'),
            drange=[-1, 1], grid_size=grid_size)

        if save_gif_name is None:
            save_gif_name = f'fakes_{cur_nimg // 1000:06d}.gif'

        n_shape = 10  # we only save 10 shapes to check performance
        if cur_tick % min((image_snapshot_ticks * 20), 100) == 0:
            save_3d_shape(mesh_v_list[:n_shape], mesh_f_list[:n_shape], run_dir, cur_nimg // 100)


def save_dense_textured_mesh_for_inference(
        G_ema, grid_z, grid_c, run_dir, save_mesh_dir=None,
        grid_tex_z=None, use_style_mixing=False):
    '''
    Generate texture mesh for generation
    :param G_ema: GET3D generator
    :param grid_z: a grid of latent code for geometry
    :param grid_c: None
    :param run_dir: save path
    :param save_mesh_dir: path to save generated mesh
    :param c_to_compute_w_avg: None
    :param grid_tex_z: latent code for texture
    :param use_style_mixing: whether we use style mixing or not
    :return:
    '''
    with torch.no_grad():
        G_ema.update_w_avg()
        save_mesh_idx = 0
        mesh_dir = os.path.join(run_dir, save_mesh_dir)
        os.makedirs(mesh_dir, exist_ok=True)
        for idx in range(len(grid_z)):
            geo_z = grid_z[idx]
            c = grid_c[idx]
            if grid_tex_z is None:
                z = grid_z[idx]
            else:
                z = grid_tex_z[idx]
            generated_mesh = G_ema.generate_dense_mesh(
                z=z, geo_z=geo_z, c=c, noise_mode='const',
                generate_raw=True, with_texture=True, truncation_psi=0.7)
                
            for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
                savemeshtes2(
                    mesh_v.data.cpu().numpy(),
                    all_uvs.data.cpu().numpy(),
                    mesh_f.data.cpu().numpy(),
                    all_mesh_tex_idx.data.cpu().numpy(),
                    os.path.join(mesh_dir, '%07d.obj' % (save_mesh_idx))
                )
                lo, hi = (-1, 1)
                img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
                img = (img - lo) * (255 / (hi - lo))
                img = img.clip(0, 255)
                mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
                mask = (mask <= 3.0).astype(np.float)
                kernel = np.ones((3, 3), 'uint8')
                dilate_img = cv2.dilate(img, kernel, iterations=1)
                img = img * (1 - mask) + dilate_img * mask
                img = img.clip(0, 255).astype(np.uint8)
                PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                    os.path.join(mesh_dir, '%07d.png' % (save_mesh_idx)))
                save_mesh_idx += 1


def save_textured_mesh_for_inference(
        G_ema, grid_z, grid_c, run_dir, save_mesh_dir=None,
        c_to_compute_w_avg=None, grid_tex_z=None, use_style_mixing=False):
    '''
    Generate texture mesh for generation
    :param G_ema: GET3D generator
    :param grid_z: a grid of latent code for geometry
    :param grid_c: None
    :param run_dir: save path
    :param save_mesh_dir: path to save generated mesh
    :param c_to_compute_w_avg: None
    :param grid_tex_z: latent code for texture
    :param use_style_mixing: whether we use style mixing or not
    :return:
    '''
    with torch.no_grad():
        G_ema.update_w_avg(c_to_compute_w_avg)
        save_mesh_idx = 0
        mesh_dir = os.path.join(run_dir, save_mesh_dir)
        os.makedirs(mesh_dir, exist_ok=True)
        for idx in range(len(grid_z)):
            geo_z = grid_z[idx]
            if grid_tex_z is None:
                tex_z = grid_z[idx]
            else:
                tex_z = grid_tex_z[idx]
            generated_mesh = G_ema.generate_dense_mesh(
                z=tex_z, geo_z=geo_z, c=None, truncation_psi=0.7,
                use_style_mixing=use_style_mixing)
            for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
                savemeshtes2(
                    mesh_v.data.cpu().numpy(),
                    all_uvs.data.cpu().numpy(),
                    mesh_f.data.cpu().numpy(),
                    all_mesh_tex_idx.data.cpu().numpy(),
                    os.path.join(mesh_dir, '%07d.obj' % (save_mesh_idx))
                )
                lo, hi = (-1, 1)
                img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
                img = (img - lo) * (255 / (hi - lo))
                img = img.clip(0, 255)
                mask = np.sum(img.astype(np.float), axis=-1, keepdims=True)
                mask = (mask <= 3.0).astype(np.float)
                kernel = np.ones((3, 3), 'uint8')
                dilate_img = cv2.dilate(img, kernel, iterations=1)
                img = img * (1 - mask) + dilate_img * mask
                img = img.clip(0, 255).astype(np.uint8)
                PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                    os.path.join(mesh_dir, '%07d.png' % (save_mesh_idx)))
                save_mesh_idx += 1


def save_geo_for_inference(G_ema, run_dir):
    '''
    Generate the 3D objs (without texture) for generation
    :param G_ema: GET3D Generation
    :param run_dir: save path
    :return:
    '''
    import kaolin as kal
    def normalize_and_sample_points(mesh_v, mesh_f, kal, n_sample, normalized_scale=1.0):
        vertices = mesh_v.cuda()
        scale = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
        mesh_v1 = vertices / scale * normalized_scale
        mesh_f1 = mesh_f.cuda()
        points, _ = kal.ops.mesh.sample_points(mesh_v1.unsqueeze(dim=0), mesh_f1, n_sample)
        return points

    with torch.no_grad():
        use_style_mixing = True
        truncation_phi = 1.0
        mesh_dir = os.path.join(run_dir, 'gen_geo_for_eval_phi_%.2f' % (truncation_phi))
        surface_point_dir = os.path.join(run_dir, 'gen_geo_surface_points_for_eval_phi_%.2f' % (truncation_phi))
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(surface_point_dir, exist_ok=True)
        n_gen = 1500 * 5  # We generate 5x of test set here
        i_mesh = 0
        for i_gen in tqdm(range(n_gen)):
            geo_z = torch.randn(1, G_ema.z_dim, device=G_ema.device)
            generated_mesh = G_ema.generate_3d_mesh(
                geo_z=geo_z, tex_z=None, c=None, truncation_psi=truncation_phi,
                with_texture=False, use_style_mixing=use_style_mixing)
            for mesh_v, mesh_f in zip(*generated_mesh):
                if mesh_v.shape[0] == 0: continue
                save_obj(mesh_v.data.cpu().numpy(), mesh_f.data.cpu().numpy(), os.path.join(mesh_dir, '%07d.obj' % (i_mesh)))
                points = normalize_and_sample_points(mesh_v, mesh_f, kal, n_sample=2048, normalized_scale=1.0)
                np.savez(os.path.join(surface_point_dir, '%07d.npz' % (i_mesh)), pcd=points.data.cpu().numpy())
                i_mesh += 1


def compute_kpts(images, pose_model, det_model, mode, vis=False, outdir=None, save_idx=0):    
    from mmdet.apis import init_detector, inference_detector
    from mmpose.apis import (init_pose_model, process_mmdet_results,
                            inference_top_down_pose_model)
    import matplotlib.pyplot as plt
    
    kps = []
    failed = 0
    images = images.detach().cpu().numpy()
    for i in range(images.shape[0]):
        img = images[i].transpose(1, 2, 0)
        img = np.clip(img[..., ::-1], 0, 255).astype("uint8")

        mmdet_results = inference_detector(det_model, img)
        person_results = process_mmdet_results(mmdet_results, cat_id=1)

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=0.3,
            format='xyxy',
            dataset=pose_model.cfg.data.test.type)

        # no human even detected:
        if len(pose_results) == 0:
            kps.append(0 * np.ones((16, 3)))
            failed = 1
        else:
            kps.append(pose_results[0]['keypoints'])

        # debug test to see how accurate kpt estimation is
        if vis:
            if outdir is not None:
                outdir_ = outdir
            else:
                outdir_ = '.'
            plt.figure(figsize=(10, 10))
            plt.imshow(img[..., ::-1])
            pose = pose_results[0]["keypoints"]
            plt.scatter(pose[:, 0], pose[:, 1])
            for j in range(len(pose)):
                plt.text(pose[j, 0], pose[j, 1], f"{pose[j, 2]:.4f}", color="red")
            plt.savefig('{}/test_{}_{}.png'.format(outdir_, mode, save_idx+i))
            print('{}/test_{}_{}.png'.format(outdir_, mode, save_idx+i))
            plt.close()
    return np.stack(kps, 0), failed