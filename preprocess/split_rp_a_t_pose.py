import os
import shutil

def main():
    src_dir = SRC_PATH
    a_pose_subdirs = []
    t_pose_subdirs = []
    for (root, subdir, files) in os.walk(src_dir):

        if subdir == []:
            if root[-1] == 'a':
                # print(root)
                a_pose_subdirs.append(root)
            elif root[-1] == 't':
                t_pose_subdirs.append(root)
            else:
                raise Exception('ERROR!!!!!!!!!')
    
    for src_path in a_pose_subdirs:
        dst_path = src_path.replace('rp_scaled_rigged_adults_2022-11-08', 'rp_scaled_a_pose_adults_2022-12-09')
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copytree(src_path, dst_path)

    src_json_path = os.path.join(src_dir, 'aligned_camera_pose_smpl.json')
    dst_json_path = src_json_path.replace('rp_scaled_rigged_adults_2022-11-08', 'rp_scaled_a_pose_adults_2022-12-09')
    shutil.copyfile(src_json_path, dst_json_path)
    
    for src_path in t_pose_subdirs:
        dst_path = src_path.replace('rp_scaled_rigged_adults_2022-11-08', 'rp_scaled_t_pose_adults_2022-12-09')
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copytree(src_path, dst_path)

    dst_json_path = src_json_path.replace('rp_scaled_rigged_adults_2022-11-08', 'rp_scaled_t_pose_adults_2022-12-09')
    shutil.copyfile(src_json_path, dst_json_path)

    return
if __name__ == "__main__":
    main()
