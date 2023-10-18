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
PORT=16003
echo "current port: ${PORT}"
NUM_GPUS=8

python -m torch.distributed.launch --use_env \
--nnodes=${ARNOLD_WORKER_NUM} \
--node_rank=${ARNOLD_ID} \
--nproc_per_node=${NUM_GPUS} \
--master_addr=${METIS_WORKER_0_HOST} \
--master_port=${PORT} \
train_3d_dist.py \
--data=datasets/THuman2.0/THuman2.0_res512  \
--gpus=${NUM_GPUS} --batch=32 --batch-gpu=4 --mbstd-group=4 --gamma=10 --dmtet_scale=2 \
--one_3d_generator=1  --fp32=0  \
--img_res=512 --norm_interval=1 \
--dis_pose_cond=True  \
--normal_dis_pose_cond=True \
--eik_weight=1e-3  \
--unit_2norm=True \
--use_normal_offset=False \
--blur_rgb_image=False \
--blur_normal_image=False \
--camera_type=blender \
--load_normal_map=True \
--with_sr=True \
--outdir=thuman_512_ckpts

