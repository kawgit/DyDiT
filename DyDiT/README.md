## 💥 Overview
![motivation](assets/motivation.png)
(a) The loss difference between DiT-S and DiT-XL across all diffusion timesteps (T = 1000). The difference is slight at most timesteps.

(b) Loss maps (normalized to the range [0, 1]) at different timesteps, show that the noise in different patches has varying levels of difficulty to predict. 

(c) Difference of the inference paradigm between the static DiT and the proposed DyDiT

![model](assets/model.png)
Overview of the proposed dynamic diffusion transformer (DyDiT). It reduces the
computational redundancy in DiT from both timestep and spatial dimensions.

## 🔨 Install

We provide an environment.yml file to help create the Conda environment in our experiments. Other environments may also works well.

```
git clone https://github.com/NUS-HPC-AI-Lab/Dynamic-Diffusion-Transformer.git
conda env create -f environment.yml
conda activate DyDiT
```


## ⚙️ Inference
Currently, we provide a pre-trained checkpoint of DyDiT $\lambda=0.7$.
| model                     |FLOPs (G) | FID    | download    
|-------------------------------|-|-----------|-----------
| DiT | 118.69 | 2.27 | - 
|DyDiT $\lambda=0.7$| 84.33 |  2.12 | [🤗](https://huggingface.co/heisejiasuo/DyDiT/resolve/main/dydit_0.7.pth?download=true)
|DyDiT $\lambda=0.5$| - |  - | in progress


Run sample_0.7.sh to sample images and evaluate the performance.
```
bash  sample_0.7.sh
```

The sample_ddp.py script which samples 50,000 images in parallel. It generates a folder of samples as well as a .npz file which can be directly used with [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and other metrics.  Please follow its instructions to download the reference batch VIRTUAL_imagenet256_labeled.npz.













## 🤔 Cite DyDiT
If you found our work useful, please consider citing us.
```
@article{zhao2024dynamic,
  title={Dynamic diffusion transformer},
  author={Zhao, Wangbo and Han, Yizeng and Tang, Jiasheng and Wang, Kai and Song, Yibing and Huang, Gao and Wang, Fan and You, Yang},
  journal={arXiv preprint arXiv:2410.03456},
  year={2024}
}
```

## ☎️ Contact
If you're interested in collaborating with us, feel free to reach out via email at wangbo.zhao96@gmail.com.

