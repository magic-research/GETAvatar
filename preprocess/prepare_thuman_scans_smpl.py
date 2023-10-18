"""
Normalize THUmanv2 scans with SMPL parameters for rendering.

THUmanv2 scans dataset
---------------------------
|----THuman2.0_Release
    |----0000
        |----0000.obj
        |----material0.jpeg
        |----material0.mtl
    |----0001     
        |----...  
    |----0525   
        |----...
THUmanv2 SMPL format
---------------------------
datasets
|----THuman2.0_smpl
    |----0000_smpl.pkl
    |----0001_smpl.pkl
    |----...
    |----0525_smpl.pkl
Run as `python3 prepare_thuman_scans_smpl.py`
"""

import os
import numpy as np
import argparse
from tqdm import tqdm
import trimesh
import pandas


# The training code normalizes f and c wrt image res -- so not doing here

def process_scans(smpl_folder, src_dir, dst_dir, index):
    """Render a latent vector interpolation video.
    """

    scan_idx = "%04d"%index
    scan_path = os.path.join(src_dir, scan_idx, scan_idx + '.obj')

    output_folder = os.path.join(dst_dir, scan_idx)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, scan_idx + '.obj')

    # load mesh
    mesh = trimesh.load(scan_path)

    # load smpl file
    pickle_path = os.path.join(smpl_folder, '%04d_smpl.pkl'%index)
    smpl_file = pandas.read_pickle(pickle_path)

    scan_faces = mesh.faces
    scan_verts = mesh.vertices

    scan_verts = scan_verts - smpl_file['transl']
    scan_verts = scan_verts / smpl_file['scale'][0]
    mesh.vertices = scan_verts

    mesh.export(output_path)


def split(a, n):
    k, m = divmod(len(a), n)
    return [ a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n) ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--tot', type=int, default=1)

    args = parser.parse_args()

    src_dir = 'datasets/THuman2.0/THuman2.0_Release'
    dst_dir = 'datasets/THuman2.0/THuman2.0_aligned_scans'
    smpl_folder = 'datasets/THuman2.0/THuman2.0_smpl'

    scan_list = sorted(os.listdir(src_dir))

    task = split(list(range(len(scan_list))), args.tot)[args.id]

    for idx in tqdm(task):
        process_scans(smpl_folder, src_dir, dst_dir, idx)

