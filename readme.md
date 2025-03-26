# DyFLUX: DyDiT for T2I generation

## Intro

We build DyFLUX by implement DyDiT on **[**FLUX-lite**](**https://huggingface.co/Freepik/flux.1-lite-8B**)**. The main modification for the architecture is adapting our TDW and SDT mechanisms to mmdit single blocks as in the following figure.

![Structure](assets/dy_flux_single_block.png)

In this repo, we release the inference code and a *work-in-progress* trained checkpoint of DyFLUX. We first trained DyFLUX based on FLUX-lite checkpoint, and then conducted a cfg-distillation to halve the actual inference batch size.

## Performance


### Geneval
| Model       | Params. (B) ↓ | s/image | FLOPs (T) ↓ | Overall ↑ | Single Object | Two Object | Counting | Colors | Position | Attribute binding |
|-------------|---------------|---------|-------------|-----------|---------------|------------|----------|--------|----------|-------------------|
| FLUX        | 12            | 18.85   | 40.8        | 66.48     | 98.75         | 84.85      | 74.69    | 76.60  | 21.75    | 42.25             |
| FLUX-Lite   | 8             | 15.23   | 30.0        | 62.06     | 98.44         | 74.24      | 64.38    | 75.53  | 17.00    | 42.75             |
| DyFLUX-Lite | 8             | 11.84   | 21.2        | 67.51     | 99.38         | 83.59      | 60.00    | 81.65  | 24.50    | 56.00             |


### User Study
We conducted a user study containing 500+ images. 12 participants were asked to rate the generated images on a scale of 1 to 5, where 5 is the most visually pleasing. Four aspects are evaluated: Instruction following, Photorealism, Aesthetic Quality and Detail Richness. 

![user_study](assets/user_study.png)

### Demos

> A glamorous young woman with long, wavy blonde hair and smokey eye makeup, posing in a luxury hotel room. She’s wearing a sparkly gold cocktail dress and holding up a white card with “Dynamic FLUX" written on it in elegant calligraphy. Soft, warm lighting creates a luxurious atmosphere.

![demo1](assets/demo1.png)


> An astronaut riding a black horse on the moon. He's wearing a Chinese badge. The background is the dark universe.

![demo2](assets/demo2.png)


> A snowman standing on the green grass, holding a sign that says 'Dynamic FLUX'.

![demo3](assets/demo3.png)

> A vibrant underwater scene with colorful coral reefs, schools of tropical fish, and a sea turtle gracefully swimming by.

![demo4](assets/demo4.png)

## Run

### Environment

```bash
git clone https://github.com/NUS-HPC-AI-Lab/Dynamic-Diffusion-Transformer.git
cd DyFLUX
conda env create -f dyflux.yml
conda activate dyflux
pip install -r requirements.txt
```

### Checkpoint Download
> 📌 Note that due to commercial licensing restrictions, we are only able to release checkpoints trained on publicly available datasets to ensure compliance while supporting academic research.

[Download DyFLUX Checkpoint](https://huggingface.co/heisejiasuo/DyFLUX/resolve/main/transformer_230000.pt?download=true)

### Inference Script

```bash
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python test_dynamic_demo.py \
--save_path YOUR_SAVE_PARH \
--prompt_path YOUR_PROMPT_PATH \
--ckpt_path YOUR_CKPT_PATH

```

Also, our method is comatible with training-free cache methods like **[TeaCache](https://github.com/ali-vilab/TeaCache)** for further accelerating:

```bash
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python test_dynamic_demo.py \
--save_path YOUR_SAVE_PARH \
--prompt_path YOUR_PROMPT_PATH \
--ckpt_path YOUR_CKPT_PATH \
--do_cache --rel_l1_thresh 0.15

```
