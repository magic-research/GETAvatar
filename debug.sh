CUDA_VISIBLE_DEVICES=7 python3 train_3d.py  \
--data=/opt/tiger/jzihang/dataset/THuman2.0/rgb_normal_dataset/thuman_scaled_2023-01-11  \
--gpus=1 --batch=4 --gamma=10  --dmtet_scale=2  --one_3d_generator=1  --fp32=0 \
--img_res=512 --dis_pose_cond=True --normal_dis_pose_cond=True --norm_interval=1 \
--eik_weight=1e-1 --blur_fade_kimg=1 --rendering_resolution_initial=512 --rendering_resolution_fade_kimg=1000 \
--unit_2norm=True \
--use_normal_offset=False \
--outdir=thu_debug