python3 train_3d.py  \
--data=/opt/tiger/jzihang/dataset/THuman2.0/rgb_normal_dataset/thuman_scaled_2023-01-11  \
--gpus=8 --batch=32 --batch-gpu=4 --gamma=10 --mbstd-group=4  --dmtet_scale=2  --one_3d_generator=1  --fp32=0 \
--img_res=512 --dis_pose_cond=True  --norm_interval=1 \
--eik_weight=1e-3 \
--outdir=checkpoints/thu_white_bg_deformation_gamma10_eik0001_0129