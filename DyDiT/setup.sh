
# Temporary quick start directions for this repo

# download trained model

wget "https://huggingface.co/heisejiasuo/DyDiT/resolve/main/dydit_0.7.pth" -O models/dydit_0.7.pth

# download tiny training dataset for testing purposes

wandb login
wget --directory-prefix=ImageNet https://image-net.org/data/tiny-imagenet-200.zip
unzip ImageNet/tiny-imagenet-200.zip