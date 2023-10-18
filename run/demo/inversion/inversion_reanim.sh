CKPT_PATH=
DATA=

POSEIDX=2355
OUTDIR=./inversion_out/pose-$POSEIDX
CUDA_VISIBLE_DEVICES=7 python3 projector_reanim.py \
--network $CKPT_PATH \
--outdir $OUTDIR \
--poseidx 2037-2040 \
--data $DATA


POSEIDX=5325
OUTDIR=./inversion_out/pose-$POSEIDX

CUDA_VISIBLE_DEVICES=7 python3 projector_reanim.py \
--network $CKPT_PATH \
--outdir $OUTDIR \
--poseidx 2037-2040 \
--data $DATA