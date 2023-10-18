import os
import json
import torch
import numpy as np
import trimesh
from tqdm import tqdm

def main(src_dir):
    subdirs = []
    for (root, subdir, files) in os.walk(src_dir):

        if subdir == []:
            subdirs.append(root)
    min_list = []
    max_list = []
    length_list = []
    height_min_list = []
    height_max_list = []

    for subdir in tqdm(subdirs):
        scan_path = os.path.join(subdir, 'mesh.obj')
        mesh = trimesh.load(scan_path)
        scan_verts = torch.tensor(np.array(mesh.vertices.astype(np.float32)))
        # print('scan_verts ', scan_verts.shape)
        body_min = torch.amin(scan_verts, dim = 0)
        body_max = torch.amax(scan_verts, dim = 0)
        print('body_min ', body_min)
        print('body_max ', body_max)
        min_list.append(1.1 * body_min - 0.1 * body_max)
        max_list.append(1.1 * body_max - 0.1 * body_min)
        length_list.append(body_max[1] - body_min[1])
        height_max_list.append(body_max[1])
        height_min_list.append(body_min[1])
        # break
    
    canonical_min = torch.amin(torch.stack(min_list, dim=0), dim=0)
    canonical_max = torch.amax(torch.stack(max_list, dim=0), dim=0)

    print('length_list ', len(length_list))
    length_min = torch.amin(torch.stack(length_list, dim=0), dim=0)
    length_max = torch.amax(torch.stack(length_list, dim=0), dim=0)

    print(canonical_min, canonical_max) 

    print('length')
    print(length_min, length_max) 

    print('height')
    height_min = torch.amin(torch.stack(height_min_list, dim=0), dim=0)
    height_max = torch.amax(torch.stack(height_max_list, dim=0), dim=0)
    print(height_min, height_max) 

    # tensor([-0.8570, -1.5028, -0.3013]) tensor([0.8693, 0.8440, 0.3145])   


if __name__ == "__main__":
    src_dir = SRC_PATH
    main(src_dir)