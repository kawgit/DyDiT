#!/bin/bash

# install stuff (assuming fundamentals are already installed)
pip install timm diffusers accelerate einops easydict termcolor wandb

# download trained model

wget --directory-prefix=models https://huggingface.co/heisejiasuo/DyDiT/resolve/main/new_release_2025_03_26/new_dydit0.7.pt

# download training dataset

curl -L -o ImageNet/imagenet-256.zip https://www.kaggle.com/api/v1/datasets/download/dimensi0n/imagenet-256
unzip ImageNet/imagenet-256.zip -d ImageNet
rm ImageNet/imagenet-256.zip