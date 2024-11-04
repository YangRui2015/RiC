import os
import json
import tqdm
import argparse
import torch
import numpy as np

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel


f = open("data/coco_test_1k.txt", 'r')
data = f.readlines()
VALIDATION_PROMPTS = [d.strip("\n") for d in data]
 

def get_samples_from_trainset(json_file, key1, key2):
    data = json.load(open(json_file))
    aes_samples = np.array([item[key1] for item in data])
    clip_samples = np.array([item[key2] for item in data])

    # normailize
    mu1 = np.mean(aes_samples)
    sigma1 = np.std(aes_samples)
    aes_samples = (aes_samples -  mu1) / sigma1
    aes_samples = np.round(aes_samples, 1)
    mu2 = np.mean(clip_samples)
    sigma2 = np.std(clip_samples)
    clip_samples = (clip_samples -  mu2) / sigma2
    clip_samples = np.round(clip_samples, 1)

    return aes_samples, clip_samples, mu1, sigma1, mu2, sigma2


def build_dataset_with_preference_n(prompts, target_rewards):
    new_prompts = []
    aes_score, clip_score = target_rewards[0], target_rewards[1] 
    for prompt in prompts:
        new_prompts.append(prompt + f"<rm1> {aes_score}" + f"<rm2> {clip_score}")

    return new_prompts


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--unet_model_name_or_path", type=str, help="unet model"
    )
    parser.add_argument(
        "--device", type=int, help="index of GPU"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    
    device = f"cuda:{args.device}"
    json_file = "data/multi_obj_with_clip_base.json"
    prompts = VALIDATION_PROMPTS
    bs = 8

    # model
    weight_dtype = torch.float32
    seed = 42
    # pretrained_model_name_or_path = "/mnt/aigc_cq/shared/txt2img_models/stable-diffusion-v1-5/"
    pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    unet_model_name_or_path = args.unet_model_name_or_path
    text_encoder = CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="text_encoder", revision=None, variant=None
            )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None)
    unet = UNet2DConditionModel.from_pretrained(
    unet_model_name_or_path, subfolder="", revision=None
    )

    # step = int(os.path.basename(os.path.dirname(unet_model_name_or_path)).split("-")[1])
    # print(f"checkpoint step: {step}")


    pipeline = StableDiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                revision=None,
                variant=None,
                safety_checker=None
            )
    pipeline = pipeline.to(device)
    pipeline.torch_dtype = weight_dtype
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_xformers_memory_efficient_attention()

    # inference under several preferences
    prompts_with_preference = prompts
    # prompts_with_preference = build_dataset_with_preference_n(prompts, target_rewards)
    prompts_with_preference_batch = [prompts_with_preference[i:min(i+bs, len(prompts_with_preference))] for i in range(0, len(prompts_with_preference), bs)]
    

    output_dir = f"results/baseline"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    idx = 0
    for i in tqdm.tqdm(range(len(prompts_with_preference_batch))):
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        with torch.autocast("cuda"):
            images = pipeline(prompts_with_preference_batch[i], num_inference_steps=50, generator=generator, num_images_per_prompt=1, guidance_scale=7.5).images
        
        for j in range(len(images)):
            images[j].save(os.path.join(output_dir, f"img-{str(idx).zfill(4)}.png"))
            idx += 1