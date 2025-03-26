import torch

from utils import combine_images_horizontally, append_line_to_file, load_txt_to_list
from flux_models.transformer_flux_dyn import DynFluxTransformer2DModel
import os
from tqdm.auto import tqdm
import argparse
from teacache_forward_utils import teacache_forward
from flux_models.pipeline_flux_dyn import FluxPipeline

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--save_path")
    parser.add_argument("--pretrained_model_name_or_path", default=None)
    parser.add_argument("--cfg_separate", action='store_true')
    parser.add_argument("--guidance_scale", type=float, default=3.5)

    parser.add_argument("--do_cache", action='store_true')
    parser.add_argument("--rel_l1_thresh", type=float, default=0.15)

    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

args = parse_args()

if __name__ == "__main__":
    # model_id = "Freepik/flux.1-lite-8B"
    torch_dtype = torch.bfloat16
    device = "cuda"

    # Load the pipe

    print('start building transformer')
    if args.do_cache:
        DynFluxTransformer2DModel.forward = teacache_forward
    transformer = DynFluxTransformer2DModel(guidance_embeds=True,
                                            num_layers=8,
                                            do_token_select=True).eval().to(dtype=torch_dtype)

    ckpt_path = args.ckpt_path
    paras = torch.load(ckpt_path, map_location='cpu')
    missing_keys, unexpected_keys = transformer.load_state_dict(
        paras["module"], strict=False
    )

    if 'latest' in ckpt_path:
        initial_global_step = paras['step']
        global_step = initial_global_step
    else:
        initial_global_step = int(ckpt_path.split("/")[-1][:-3].split("_")[-1])
        global_step = initial_global_step

    print(f"[INFO] successfully resumed from {args.ckpt_path}, missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")


    print('start building pipe')
    
    if args.pretrained_model_name_or_path is not None:
        pipe = FluxPipeline.from_pretrained(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=torch_dtype
        ).to(device)
    else:
        pipe = FluxPipeline.from_pretrained(
            "Freepik/flux.1-lite-8B", 
            transformer=transformer,
            torch_dtype=torch_dtype
        ).to(device)

    # Inference
    guidance_scale = args.guidance_scale 
    n_steps = 28
    seed = 42


    if args.do_cache:
        pipe.transformer.__class__.enable_teacache = True
        pipe.transformer.__class__.cnt = 0
        pipe.transformer.__class__.num_steps = n_steps
        pipe.transformer.__class__.rel_l1_thresh = float(args.rel_l1_thresh)
        print("thresh=", float(args.rel_l1_thresh))
        pipe.transformer.__class__.accumulated_rel_l1_distance = 0
        pipe.transformer.__class__.previous_modulated_input = None
        pipe.transformer.__class__.previous_residual = None
        save_root_path = args.save_path + f"_cachethresh_{args.rel_l1_thresh}"
    else:
        save_root_path = args.save_path 
    
    save_root_path = f"{save_root_path}_guidance_{guidance_scale}"
    
    if args.cfg_separate:
        save_root_path = f"{save_root_path}_separate_cfg"

    if args.prompt_path is not None:
        prompts = load_txt_to_list(args.prompt_path)
    else:
        prompts = ["An astronaut riding a black horse on the moon. He's wearing a Chinese badge. The background is the dark universe."]

    img_paths = []

    os.makedirs(save_root_path, exist_ok=True)
    pipe.set_progress_bar_config(disable=True)
    for i in tqdm(range(len(prompts))):
        prompt = prompts[i]
        # prompt_cn = prompts_cn[i]
        print(f"{prompt}")
        with torch.inference_mode():
            res = pipe(
                prompt=prompt,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                num_inference_steps=n_steps,
                guidance_scale=guidance_scale,
                height=1024,
                width=1024,
                num_images_per_prompt=args.num_images_per_prompt,
                max_sequence_length=256
            )
        save_img_path = f"{save_root_path}/image_{i:04d}"
        os.makedirs(save_img_path, exist_ok=True)
        img_paths_i = []
        for j in range(len(res[0])):
            img_path = f"{save_img_path}/{j:04d}.png"
            img_paths.append(img_path)
            img_paths_i.append(img_path)
            res[0][j].save(img_path)
        
        if len(img_paths_i) > 1:
            combined_img_i = combine_images_horizontally(img_paths_i)
            combined_img_i.save(f"{save_img_path}/image_{i:04d}.png")
        
        append_line_to_file(file_path=f'{save_img_path}/prompt.txt', line_to_append=prompt)


