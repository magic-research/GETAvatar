# Print env variable info
echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

# NCCL debugging flag on
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0

export BYTED_TORCH_C10D_LOG_LEVEL=WARNING

PORTS=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# PORT=${PORTS[0]}
PORT=16003
echo "current port: ${PORT}"
NUM_GPUS=8

torchrun \
--nnodes=${ARNOLD_WORKER_NUM} \
--node_rank=${ARNOLD_ID} \
--nproc_per_node=${NUM_GPUS} \
--master_addr=${METIS_WORKER_0_HOST} \
--master_port=${PORT} \
dist_train_3d.py \
--data=/opt/tiger/jzihang/dataset/Renderpeople/rgb_normal_dataset/rp_rigged_and_posed_1024x1024_2023-02-06  \
--gpus=${NUM_GPUS} --batch=32 --batch-gpu=4 --mbstd-group=4 --gamma=20 --dmtet_scale=2 \
--one_3d_generator=1  --fp32=0  \
--img_res=512 --norm_interval=0 \
--dis_pose_cond=True  \
--normal_dis_pose_cond=True \
--eik_weight=1e-3  \
--unit_2norm=True \
--use_normal_offset=False \
--blur_rgb_image=False \
--blur_normal_image=False \
--camera_type=blender \
--load_normal_map=False \
--with_sr=True \
--part_disc=False \
--outdir=checkpoints/rp_res512_no_norm

python3 /home/tiger/code/Avatargen_eg3d/run/scripts/occ.py

# --outdir=checkpoints/thu_dist_res512_eik0001_0209_wo_norm
