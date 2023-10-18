import os
import torch
import numpy as np
import json

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def convert_json(src_dir, src_json_name, dst_json_name, device='cpu'):
    with open(os.path.join(src_dir, src_json_name)) as f:
        labels = json.load(f)['labels']
    
    transforms = {}
    for fname, label in labels.items():
        """
            label info: 
                0:16    cam2world_matrix    shape (16,)
                16:25   intrinsics          shape (9,)
                25:28   global_orient       shape (3,)
                28:97   body_pose           shape (69,)
                97:107  betas               shape (10,)
        """
        c_cam_smpl = torch.tensor(np.array(label).astype(np.float32)).float().to(device)
        cam2world_matrix_tensor = c_cam_smpl[:16].view(4, 4)
        world2cam_matrix_tensor = torch.linalg.inv(cam2world_matrix_tensor)
        world2cam_matrix_list =listify_matrix(world2cam_matrix_tensor)
        extrinsics_list = [element for row in world2cam_matrix_list for element in row]

        for idx, item in enumerate(extrinsics_list):
            extrinsics_list[idx] = float(item)

        """
        label info: 
            0:16    world2camera_matrix shape (16,)
            16:19   global_orient       shape (3,)
            19:88   body_pose           shape (69,)
            88:98  betas                shape (10,)
        """
        smpl_param_list = label[25:]
        extrinsics_smpl_list = extrinsics_list + smpl_param_list

        assert len(extrinsics_smpl_list)==98
        transforms[fname] = extrinsics_smpl_list

        # print('extrinsics_smpl_list')
        # print(extrinsics_smpl_list)
        # break

    data = {'labels': transforms}

    with open(os.path.join(src_dir, dst_json_name), 'w') as out_file:
        json.dump(data, out_file, indent=4)

    return

if __name__ == "__main__"
    # convert_json(src_dir, src_json_name, dst_json_name)
    src_dir = DATA_PATH
    src_json_name = 'aligned_camera_pose_smpl.json'
    dst_json_name = 'extrinsics_smpl.json'
    if os.path.exists(src_dir):
        convert_json(src_dir, src_json_name, dst_json_name)
    else:
        print(os.path.basename(src_dir) + ' do not exist!')

    src_dir = SRC_PATH
    
    if os.path.exists(src_dir):
        convert_json(src_dir, src_json_name, dst_json_name)
    else:
        print(os.path.basename(src_dir) + ' do not exist!')
