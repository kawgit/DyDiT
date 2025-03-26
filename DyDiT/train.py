# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import os
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import misc as misc
from util.logger import create_logger
from datasets.image_datasets import build_image_dataset
from easydict import EasyDict
from loss import DynamicLoss
import wandb
from utils import clip_grad_norm_
# from util import compute_neuron_head_importance
#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """

    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    misc.init_distributed_mode(args)
    # Setup DDP:
    # dist.init_process_group("nccl")
    assert args.global_batch_size % misc.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = misc.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * misc.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    if rank == 0 and args.wandb:
        wandb.init(
        # set the wandb project where this run will be logged
        project="imagenet",
        name=f'ourmodel_{args.model}_withdynamictoken_{args.token_ratio}')
        
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    logger = create_logger(output_dir=args.results_dir, dist_rank=misc.get_rank(), name=f"{args.model}_{int(time())}")
    logger.info(f"Experiment directory created at {args.results_dir}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={misc.get_world_size()}.")

    # args.data_path = DATASETS
    # args.data_path = DATASETS
    dataset = build_image_dataset(args)
    logger.info("dataset length {}".format(len(dataset)))
    sampler = DistributedSampler(
        dataset,
        num_replicas=misc.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // misc.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    args.model_depth = model.depth
    ckpt_path = args.ckpt
    checkpoint_model = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if args.resume:
        resumed_model = checkpoint_model['model']
    else:
        resumed_model = checkpoint_model
    
    state_dict = model.state_dict()
    
    for k in ['y_embedder.embedding_table.weight']:
        if k in state_dict and resumed_model[k].shape != state_dict[k].shape:
            logger.info(f"Removing key {k} from pretrained checkpoint")
            del resumed_model[k]

    msg = model.load_state_dict(resumed_model, strict=False)
    logger.info(msg)

    # for name, p in model.named_parameters():
    #     if name in msg.missing_keys:
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False
    # for name, p in model.named_parameters():
    #     if name in msg.missing_keys:
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of trainable params (M): %.2f' % (n_parameters / 1.e6))
    
    
    model_without_ddp = model   
    
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model_without_ddp).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    if args.distributed:
        model = DDP(model_without_ddp.to(device), device_ids=[rank])
    else:
        model = model_without_ddp.to(device)
    
    

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(f"/path/stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")


    # head_importance, neuron_importance = compute_neuron_head_importance(args, vae, diffusion, loader, device)
    # reorder_neuron_head(args, head_importance, neuron_importance)


    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    logger.info(f"learning rate set to {args.lr}")
    if args.resume:
        logger.info(f'resume optimizer from {ckpt_path}')
        resumed_optimizer = checkpoint_model['opt']
        opt.load_state_dict(resumed_optimizer)  


    # logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module if args.distributed else model, decay=0)  # Ensure EMA is initialized with synced weights
    if args.resume:
        logger.info(f'resume ema from {ckpt_path}')
        resumed_ema = checkpoint_model['ema']
        ema.load_state_dict(resumed_ema)

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode


    select_config = EasyDict(        
        token_ratio=2.,
        token_target_ratio=args.token_ratio,
        token_minimal=0.,
        token_minimal_weight=0.,        
        ) 
    token_loss_func = DynamicLoss(
        token_target_ratio=select_config.token_target_ratio,
        token_loss_ratio=select_config.token_ratio,
        token_minimal=select_config.token_minimal,
        token_minimal_weight=select_config.token_minimal_weight,
        model_name=args.model,
        model_depth=args.model_depth)
    
    
    # Variables for monitoring/logging purposes:
    train_steps = 0
    if args.resume:
        train_steps = int(args.ckpt.split("/")[-1].split(".")[0])
    log_steps = 0
    running_loss = 0
    running_dynamic_loss = 0
    running_diffusion_loss = 0
    running_complete_diffusion_loss = 0
    running_distill_loss = 0 
    # running_activate_rate = 0
    start_time = time()
    flag = False
    t_sampling_choice = [0, 250, 500, 750, 1000]
    range_choices = torch.arange(int(args.global_batch_size // misc.get_world_size()), device=device) % 4
    lower_bounds = torch.tensor(t_sampling_choice[:-1], device=device)[range_choices]

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            if args.t_sample:
                t = torch.randint(0, 250, (x.shape[0],), device=device) + lower_bounds
            else:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)


            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y)

            if args.warmup and (train_steps <= args.warmup_step):
                flops_target_ratio = 1.0 - (1.0 - args.token_ratio) * train_steps / args.warmup_step
            else:
                flops_target_ratio = args.token_ratio
                
            if train_steps < 100:
                flops_target_ratio = 1.0

    
            token_loss_func.token_target_ratio = flops_target_ratio



            loss_dict, attn_channel_masks, mlp_weight_masks, masks = diffusion.training_losses(model, x, t, model_kwargs)
            dynamic_loss, real_activate_rate = token_loss_func(attn_channel_masks, mlp_weight_masks, masks)
            diffusion_loss = loss_dict["loss"].mean()
            complete_diffusion_loss = (loss_dict["complete_mse"] + loss_dict["complete_vb"]).mean()
            
            total_loss = diffusion_loss + complete_diffusion_loss + dynamic_loss
            
            opt.zero_grad()
            total_loss.backward()

            opt.step()
            gradient_norm_dyn = clip_grad_norm_(model.parameters(), max_norm=args.clip_max_norm, clip_grad=True)
            if gradient_norm_dyn < args.clip_max_norm:
                opt.step()
            else:
                logger.info(f"Step {train_steps}: Skipping update due to large gradient norm {gradient_norm_dyn}")


            update_ema(ema, model.module if args.distributed else model)

            # Log loss values:
            running_loss += total_loss.item()
            running_dynamic_loss += dynamic_loss.item()
            running_diffusion_loss += diffusion_loss.item()
            running_complete_diffusion_loss += complete_diffusion_loss.item()
            # running_activate_rate += real_activate_rate.item()

            
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_dynamic_loss = torch.tensor(running_dynamic_loss / log_steps, device=device)
                avg_diffusion_loss = torch.tensor(running_diffusion_loss / log_steps, device=device)
                avg_complete_diffusion_loss = torch.tensor(running_complete_diffusion_loss / log_steps, device=device)
                # avg_activate_rate = torch.tensor(running_activate_rate / log_steps, device=device)

                
                
                avg_loss = misc.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_dynamic_loss = misc.all_reduce(avg_dynamic_loss, op=dist.ReduceOp.SUM)
                avg_diffusion_loss = misc.all_reduce(avg_diffusion_loss, op=dist.ReduceOp.SUM)
                avg_complete_diffusion_loss = misc.all_reduce(avg_complete_diffusion_loss, op=dist.ReduceOp.SUM)

       
                
                
                avg_loss = avg_loss.item() / misc.get_world_size()
                avg_dynamic_loss = avg_dynamic_loss.item() / misc.get_world_size()
                avg_diffusion_loss = avg_diffusion_loss.item()  / misc.get_world_size()
                avg_complete_diffusion_loss = avg_complete_diffusion_loss.item()  / misc.get_world_size()
            
    
                if rank == 0 and args.wandb:
                    wandb.log({"Train Loss": avg_loss, "Diffusion Loss": avg_diffusion_loss, "Dynamic Loss": avg_dynamic_loss})
                    
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} Diffusion Loss: {avg_diffusion_loss:.4f} Complete Diffusion Loss: {avg_complete_diffusion_loss:.4f} Dynamic Loss: {avg_dynamic_loss:.4f} FLOPS Ratio: {flops_target_ratio:.4f} Real Activate Ratio: {real_activate_rate.item():.4f} Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables: 
                running_loss = 0
                running_diffusion_loss = 0
                running_complete_diffusion_loss = 0
                running_dynamic_loss = 0
                running_distill_loss = 0
                # running_activate_rate = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if (train_steps == args.total_train_steps) or (train_steps % args.ckpt_every == 0 and train_steps > 0):
                if rank == 0:
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{args.results_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            if train_steps == args.total_train_steps:
                flag = True
                break
        if flag:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    wandb.finish()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/earth-nas/datasets/imagenet-1k/train/")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--token_ratio", type=float, default=0.5)
    parser.add_argument("--total_train_steps", type=int, default=150000) # DiffFit: total_batchsize=256, total_training_steps=24k
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--wandb", action='store_true', default=False)
    parser.add_argument("--t_sample", action='store_true', default=False)
    parser.add_argument("--warmup", action='store_true', default=False)
    parser.add_argument("--warmup_step", type=int, default=1000)
    parser.add_argument("--clip_max_norm", type=float, default=15.0)
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
