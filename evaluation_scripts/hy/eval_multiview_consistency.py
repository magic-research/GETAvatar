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
import argparse
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



def gen_interp_video(G, D, mp4: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, device=torch.device('cuda'), label_pool='', outdir='', **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    # config = json.load(open('/home/tiger/Code/morphable_nerf/im3dmm/morphable_model/mpi_flame/flame_config.json', 'r'))     
    # t_args = argparse.Namespace()
    # t_args.__dict__.update(config)
    # flame = FLAME(t_args).cuda(device)

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

    # all_seeds = [0, 1, 2, 4]
    all_seeds = seeds
    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
           
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)

    feature_dim = G.rendering_kwargs.get('feature_dim', 0)
    if cfg == 'ffhq':
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    else:
        intrinsics = torch.tensor([[1015./224, 0.0, 0.5],[0.0, 1015./224, 0.5], [0.0, 0.0, 1.0]], device=device)

    frame_3dmms = np.load(label_pool)
    exp_seed = np.random.randint(0, len(frame_3dmms))
    frame_3dmm = frame_3dmms[exp_seed, -feature_dim:]

    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)

    ws_bcg=None
    
    total_frames = 512
    # for frame_idx in tqdm(range((num_frames // 2 - 8) * n_interp)):
    pixels = []
    for frame_idx in tqdm(range(total_frames)):
        imgs = []

        features_3dmm = torch.from_numpy(frame_3dmm).to(device).float()
        # features_3dmm = torch.zeros_like(features_3dmm)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), features_3dmm.reshape(-1, feature_dim)], 1)
        conditioning_params = conditioning_params.repeat([len(zs), 1])

        ws = G.mapping(zs, conditioning_params, truncation_psi=psi, truncation_cutoff=truncation_cutoff)

        if ws_bcg is None:
            ws_bcg = ws.clone()
        pitch_range = 0 
        yaw_range = 3.14/8 
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw_range * np.cos(3.14 * min(frame_idx, total_frames-1) / total_frames),
                                                np.pi/2-0.05,
                                                cam_pivot, radius=cam_radius, device=device)
        
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), features_3dmm.reshape(-1, feature_dim)], 1)

        # # d_score, p_est = run_D(D, out, camera_params.repeat([len(ws), 1]))
        img_outdir = os.path.join(outdir, 'multiview')
        if not os.path.exists(img_outdir):
            os.mkdir(img_outdir)

        for k in range(len(ws)):
            out = G.synthesis(ws[k:k+1], camera_params.repeat([len(ws[k:k+1]), 1]), ws_bcg=ws_bcg[k:k+1], noise_mode='const')
            img = out['image'][0].permute(1, 2, 0)
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # pixels.append(img[230:280, 434:435:, :])
            PIL.Image.fromarray(img.cpu().numpy(), 'RGB').save(os.path.join(img_outdir, 'seed_%02d_%04d.png' % (seeds[k], frame_idx)))
)
        video_out.append_data(layout_grid(out['image'], grid_w=grid_w, grid_h=grid_h))

    video_out.close()
 
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

    if interpolate:
        output = os.path.join(outdir, 'multiview.mp4')
        gen_interp_video(G=G, D=D, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes, label_pool=label_pool, outdir=outdir)
    else:
        for seed in seeds:
            output = os.path.join(outdir, f'{seed}.mp4')
            seeds_ = [seed]
            gen_interp_video(G=G, D=D, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds_, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, label_pool=label_pool, outdir=outdir)

def generate_tex():
    img_folder = IMG_DIR
    seed = 444
    img_paths = glob.glob(os.path.join(img_folder, 'i*.png'))
    img_paths.sort()
    img_paths = img_paths

    pixel_patch = []
    row_lo = 250
    row_hi = 300
    col = 330
    for img_path in img_paths:
        img = np.array(PIL.Image.open(img_path))
        pixel_patch.append(img[col:col+1, row_lo:row_hi, :])
    pixel_bench = np.concatenate(pixel_patch, axis=0)
    PIL.Image.fromarray(pixel_bench, 'RGB').save(os.path.join(img_folder, 'bench_teeth_%02d.png' % seed))

    pixel_patch = []
    row_lo = 225
    row_hi = 275
    col = 90
    for img_path in img_paths:
        img = np.array(PIL.Image.open(img_path))
        pixel_patch.append(img[col:col+1, row_lo:row_hi, :])
    pixel_bench = np.concatenate(pixel_patch, axis=0)
    PIL.Image.fromarray(pixel_bench, 'RGB').save(os.path.join(img_folder, 'bench_hair_%02d.png' % seed))
#----------------------------------------------------------------------------
import json
def export_datalabels():
    camera_labels = json.load(open('~/Data/neural_rendering/eg3d/dataset/dataset.json', 'r'))
    fit_labels = json.load(open('~/Data/neural_rendering/eg3d/dataset/3dmm.json', 'r'))

    camera_labels_dict={}
    for i, example in enumerate(camera_labels['labels']):
        name, label = example
        name_split = name.split('_')
        if len(name_split) > 2 or not name.endswith('_00.png'):
            continue
        camera_labels_dict[name] = label
    
    agg_labels = []
    for i, example in enumerate(fit_labels['labels']):
        name, label = example
        if not name in camera_labels_dict:
            continue
        agg_labels.append(np.concatenate([camera_labels_dict[name], label], axis = -1))
    agg_labels = np.array(agg_labels, np.float32)
    np.save('~/Data/neural_rendering/eg3d/dataset/ffhq_labels.npy', agg_labels)

if __name__ == "__main__":
    # generate_images() # pylint: disable=no-value-for-parameter
    generate_tex()
    # export_datalabels()
    
#----------------------------------------------------------------------------
