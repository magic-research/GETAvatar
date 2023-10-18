"""
Convert the normalized THUmanv2 dataset to dataset format usable for Avatargen_eg3d.
dataset format
---------------------------
datasets
|----THuman2.0_res512
    |----0000
        |----0000.png
        |----0001.png   
        |---- ...              
        |----0099.png  
        |----mesh.obj
        |----blender_transforms.json
    |----0001     
        |----...  
    |----0525   
        |----...
Run as 
python3 prepare_thuman_json.py
"""


import os
import json
import torch
import numpy as np
import pandas

# K = [[fx, s, x_0],
#      [0, fy, y_0],
#      [0, 0, 1]]

# image_res = 512         # should be res. of input image to network
# width = height = image_res
fov = 0.8575560450553894 # radians

fx = fy = 0.5 / np.tan(0.5 * fov)
cx = cy = 0.5

INTRINSIC_MTX = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

# The training code normalizes f and c wrt image res -- so not doing here
def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def generate_json(subdirs, src_dir, smpl_dir):

    transforms = {}
    subdir_list = sorted(subdirs)
    print('length ', len(subdir_list))

    for subdir in subdir_list:
        f = open(os.path.join(subdir, 'blender_transforms.json'), 'r')
        meta = json.load(f)

        scan_index = subdir.split('/')[-1]

        smpl_pickle_path = os.path.join(smpl_dir, scan_index + '_smpl.pkl')
        
        assert os.path.exists(smpl_pickle_path)

        smpl_file = pandas.read_pickle(smpl_pickle_path)

        """
            label info: 
                0:16    cam2world_matrix    shape (16,)
                16:25   intrinsics          shape (9,)
                25:28   global_orient       shape (3,)
                28:97   body_pose           shape (69,)
                97:107  betas               shape (10,)
        """

        smpl_param = np.concatenate([
                                smpl_file['global_orient'].reshape(1,-1),
                                smpl_file['body_pose'].reshape(1,-1),
                                smpl_file['betas'][:,:10]], axis=1)[0]

        smpl_param = smpl_param.tolist()
        
        for f in meta['frames']:
            file_name = os.path.relpath(f['file_path'], src_dir)
            transform_mtx_tensor = torch.Tensor(f['blender_transform_matrix']).float()
            re_correc_mtx = torch.Tensor(
                [
                    [1.0, 0.0, 0.0,  0.0],
                    [0.0, 0.0, 1.0,  0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0,  1.0]
                ]
            ).float()
            re_correc_transform_mtx_tensor = re_correc_mtx @ transform_mtx_tensor
            re_correc_transform_mtx_list = listify_matrix(re_correc_transform_mtx_tensor)
            transform_mtx = [element for row in re_correc_transform_mtx_list for element in row]
            
            for idx, item in enumerate(transform_mtx):
                transform_mtx[idx] = float(item)

            transform_mtx.extend(INTRINSIC_MTX)
            transform_mtx.extend(smpl_param)

            """
            label info: 
                0:16    cam2world_matrix    shape (16,)
                16:25   intrinsics          shape (9,)
                25:28   global_orient       shape (3,)
                28:97   body_pose           shape (69,)
                97:107  betas               shape (10,)
            """

            transforms[file_name] = transform_mtx

    data = {'labels': transforms}

    with open(os.path.join(src_dir, 'aligned_camera_pose_smpl.json'), 'w') as out_file:
        json.dump(data, out_file, indent=4)


def convert_dataset(src_dir, smpl_dir):
    """
    Args
    ----
    src_dir: Render THUman2.0 dataset directory
    smpl_dir: THUman2.0 smpl dataset directory
    """

    subdirs = []
    for (root, subdir, files) in os.walk(src_dir):

        if subdir == []:
            subdirs.append(root)

    generate_json(subdirs, src_dir, smpl_dir)

if __name__ == "__main__":
    src_dir = 'datasets/THuman2.0/THuman2.0_res512'
    smpl_dir = 'datasets/THuman2.0/THuman2.0_smpl'

    convert_dataset(src_dir, smpl_dir)