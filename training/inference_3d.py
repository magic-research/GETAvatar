# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import copy
import os

import numpy as np
import torch
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
def inference(
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
    # D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(
    #     device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()  # deepcopy can make sure they are correct.
    if resume_pretrain is not None and (rank == 0):
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G.load_state_dict(model_state_dict['G'], strict=True)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        # D.load_state_dict(model_state_dict['D'], strict=True)
    
    grid_size = (2, 4)
    n_shape = grid_size[0] * grid_size[1]
    grid_z = torch.randn([n_shape, G.z_dim], device=device).split(1)  # random code for geometry
    grid_tex_z = torch.randn([n_shape, G.z_dim], device=device).split(1)  # random code for texture

    dataset = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    labels = dataset._get_raw_labels()
    
    rand_idx = np.random.RandomState(seed=0).choice(range(len(labels)), n_shape)
    grid_c = torch.from_numpy(labels[rand_idx]).to(device).split(1)

    print('==> generate ')
    if inference_for_pck_eval:
        training_set_sampler = misc.InfiniteSampler(dataset=dataset, rank=0, num_replicas=1, seed=0, shuffle=False)
        data_iter = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=training_set_sampler, batch_size=4))
        pose_model, det_model = load_mmcv_models()
        det_model = det_model.to(device)
        pose_model = pose_model.to(device)
        pck = save_visualization_inference_for_pck_eval(
            data_iter, G_ema, batch_gen=4, pose_model=pose_model, 
            det_model=det_model, max_items=10000, device=device,
            outdir=run_dir
        )
        pck_path = os.path.join(run_dir, 'pck_results.txt')
        model_name = resume_pretrain.split('/')[-3]
        # TODO: save pck to the file
        with open(pck_path, 'w') as f:
            f.write(f"{model_name}: {pck}")
    elif inference_for_depth_eval:
        training_set_sampler = misc.InfiniteSampler(dataset=dataset, rank=0, num_replicas=1, seed=0, shuffle=False)
        data_iter = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=training_set_sampler, batch_size=1))
        os.makedirs(os.path.join(run_dir, "depth"), exist_ok=True)
        save_visualization_inference_for_depth_eval(
            data_iter, G_ema, batch_gen=1, max_items=1000, 
            device=device, outdir=run_dir
        )
    else:
        save_visualization_inference(
            G_ema, grid_z, grid_c, run_dir, 0, grid_size, 0,
            save_all=False,
            grid_tex_z=grid_tex_z
        )

    if inference_to_generate_textured_mesh:
        print('==> generate inference 3d shapes with texture')
        save_textured_mesh_for_inference(
            G_ema, grid_z, grid_c, run_dir, save_mesh_dir='texture_mesh_for_inference',
            c_to_compute_w_avg=None, grid_tex_z=grid_tex_z)

    if inference_save_interpolation:
        print('==> generate interpolation results')
        save_visualization_for_interpolation(G_ema, save_dir=os.path.join(run_dir, 'interpolation'))

    if inference_compute_fid:
        print('==> compute FID scores for generation')
        for metric in metrics:
            training_set_kwargs = clean_training_set_kwargs_for_metrics(training_set_kwargs)
            training_set_kwargs['split'] = 'test'
            result_dict = metric_main.calc_metric(
                metric=metric, G=G_ema,
                dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank,
                device=device)
            metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=resume_pretrain)

    if inference_generate_geo:
        print('==> generate 7500 shapes for evaluation')
        save_geo_for_inference(G_ema, run_dir)


def load_mmcv_models():
    from mmdet.apis import init_detector
    from mmpose.apis import init_pose_model

    dir_path = os.getcwd()
    pose_config = os.path.join(dir_path, "mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hrnet_w32_mpii_256x256_dark.py")
    pose_checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark-f1601c5b_20200927.pth"
    det_config = os.path.join(dir_path, "mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py")
    det_checkpoint = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

    # initialize pose model
    pose_model = init_pose_model(pose_config, pose_checkpoint)
    # initialize detector
    det_model = init_detector(det_config, det_checkpoint)

    return pose_model, det_model
