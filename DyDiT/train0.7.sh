#!/usr/bin/env sh


GPUS=${GPUS:-1}
PORT=$((12000 + $RANDOM % 20000))
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}


CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
       --master_addr=$MASTER_ADDR \
       --nproc_per_node=$GPUS \
       --master_port=$PORT \
       --use_env \
       train.py \
       --model "DiT-XL/2" \
       --global-batch-size 1 \
       --ckpt "models/new_dydit0.7.pt" \
       --results-dir "results0.7" \
       --t_sample \
       --warmup \
       --lr 2.5e-5 \
       --global-seed 16 \
       --token_ratio 0.7 \
       --clip_max_norm 50.0 \
       --warmup_step 5000 \
       --data-path ImageNets/tiny-imagenet-200/train \
       --ckpt-every 20


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python -m torch.distributed.launch \
#         --master_addr=$MASTER_ADDR \
#         --nproc_per_node=$GPUS \
#         --master_port=$PORT \
#         --use_env \
#         ./sample_ddp.py \
#         --model DiT-XL/2 \
#         --image-size 256  \
#         --ckpt ./results0.7/0150000.pt \
#         --sample-dir "./samples0.7/150000step" \
#         --num-fid-samples 50000 


# set -x
# python evaluator.py ../../imagenet_fid/VIRTUAL_imagenet256_labeled.npz \
# ./samples0.7/150000step.npz
# set +x





