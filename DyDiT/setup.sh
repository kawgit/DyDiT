
# Temporary quick start directions for this repo

# download trained model

wget --directory-prefix=models https://huggingface.co/heisejiasuo/DyDiT/resolve/main/new_release_2025_03_26/new_dydit0.7.pt

# download tiny training dataset for testing purposes

wget --directory-prefix=ImageNet https://image-net.org/data/tiny-imagenet-200.zip
unzip ImageNet/tiny-imagenet-200.zip
rm ImageNet/tiny-imagenet-200.zip