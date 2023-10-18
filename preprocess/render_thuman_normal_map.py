import os
import torch
import numpy as np
import argparse
import json
from tqdm import tqdm
import trimesh
import PIL.Image
from tqdm import tqdm

from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_obj
from renderer_utils import create_cameras, create_mesh_normal_renderer


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

def save_image(img, fname, drange):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    
    N, C, H, W = img.shape
    img = img.reshape([C, H, W])
    img = img.transpose(1, 2, 0)
    img = img.reshape([H, W, C])

    assert C in [1, 3]
    if not fname is None:
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)
    return img

def split(a, n):
    k, m = divmod(len(a), n)
    return [ a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n) ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--tot', type=int, default=1)
    args = parser.parse_args()

    src_dir = 'datasets/THuman2.0/THuman2.0_res512'
    json_fname = 'extrinsics_smpl.json'

    device = 'cuda'

    with open(os.path.join(src_dir, json_fname), 'r') as f:
        labels = json.load(f)['labels']

    total_rgb_img_list = []
    for rgb_fname in labels.keys():
        total_rgb_img_list.append(rgb_fname)

    spli_rgb_img_list = split(total_rgb_img_list, args.tot)[args.id]

    for idx in tqdm(range(len(spli_rgb_img_list))):
        rgb_img_fname = spli_rgb_img_list[idx]
        c = labels[rgb_img_fname]
        c_cam = c[:16]

        world2cam_matrix = torch.tensor(c_cam).view(4, 4).to(device)
        cam2world_matrix = torch.linalg.inv(world2cam_matrix)

        aligned_matrix = torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        ).float().to(device)
        world2cam_matrix = aligned_matrix @ world2cam_matrix
        cam2world_matrix = torch.linalg.inv(world2cam_matrix)

        render_R = cam2world_matrix[:3, :3]
        render_T = torch.tensor([world2cam_matrix[0, 3], world2cam_matrix[1, 3], world2cam_matrix[2, 3]]).type_as(cam2world_matrix)

        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        cameras = create_cameras(R=render_R[None], T=render_T[None], fov=fovyangle, device=device)

        img_res = int(1024)
        renderer = create_mesh_normal_renderer(
                    cameras, image_size=img_res,
                    light_location=((1.0,1.0,1.0),), specular_color=((0.2,0.2,0.2),),
                    ambient_color=((0.2,0.2,0.2),), diffuse_color=((0.65,.65,.65),),
                    device=device)

        scan_path = os.path.join(src_dir, os.path.dirname(rgb_img_fname), 'mesh.obj')

        scan_verts, scan_faces, aux = load_obj(scan_path, 
                                                device=device,
                                                load_textures=False)

        scan_faces = scan_faces.verts_idx.long()

        scan_mc_mesh = Meshes(
            verts=[scan_verts],
            faces=[scan_faces],
            textures=None
        )
        normal_img = renderer(scan_mc_mesh)[..., :3].permute(0, 3, 1, 2)

        save_normal_img = normal_img.detach().cpu().numpy()
        save_normal_path = os.path.join(src_dir, rgb_img_fname.replace('.png', '_normal.png'))

        save_image(save_normal_img, save_normal_path, drange=[-1, 1])

