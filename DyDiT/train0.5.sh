#!/usr/bin/env sh


GPUS=${GPUS:-8}
PORT=$((12000 + $RANDOM % 20000))
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

# stage1 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
       --master_addr=$MASTER_ADDR \
       --nproc_per_node=$GPUS \
       --master_port=$PORT \
       --use_env \
       train.py \
       --model "DiT-XL/2" \
       --global-batch-size 256 \
       --ckpt "/path/dit_xl_256.pt" \
       --results-dir "results0.5" \
       --t_sample \
       --warmup \
       --global-seed 16 \
       --lr 1e-4 \
       --token_ratio 0.5 \
       --clip_max_norm 200.0 \
       --ckpt-every 3000


# stage 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
       --master_addr=$MASTER_ADDR \
       --nproc_per_node=$GPUS \
       --master_port=$PORT \
       --use_env \
       train_without_completemodel.py \
       --model "DiT-XL/2" \
       --global-batch-size 256 \
       --ckpt "./results0.5/0030000.pt" \
       --results-dir "results0.5" \
       --t_sample \
       --warmup \
       --global-seed 16 \
       --lr 1e-4 \
       --token_ratio 0.5 \
       --clip_max_norm 200.0 \
       --ckpt-every 5000 \
       --resume \
       --total_train_steps 200000






CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        --use_env \
        ./sample_ddp.py \
        --model DiT-XL/2 \
        --image-size 256  \
        --ckpt ./results0.5/0200000.pt \
        --sample-dir "./samples0.5/200000step" \
        --num-fid-samples 50000 


set -x
python evaluator.py ../../imagenet_fid/VIRTUAL_imagenet256_labeled.npz \
./samples0.5/200000step.npz
set +x






