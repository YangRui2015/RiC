import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

pretrained_model_name_or_path = "/mnt/aigc_cq/shared/txt2img_models/stable-diffusion-v1-5"

device = "cuda"
weight_dtype = torch.float32

seed = 42

text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=None, variant=None
        )
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None)
unet = UNet2DConditionModel.from_pretrained(
pretrained_model_name_or_path, subfolder="unet", revision=None
)

pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=None,
            variant=None,
        )

# Run a final round of inference.
images = []
# validation_prompts = [
#     "Mountains range with waterfall, purple haze, art by greg rutkowski and magali villeneuve, artstation.",
#     "Portrait of a female elf warlock, long pointy ears, glowing green eyes, bushy red hair and freckles + medieval setting, detailed face, highly detailed, digital painting, artstation.",
#     "An concept art illustration, photorealistic tribal people working, fantasy street with huts, large insect and plant biomes, ultra realistic, style by wlop.",
#     "A half - masked rugged laboratory engineer man with cybernetic enhancements as seen from a distance, scifi character portrait by greg rutkowski, esuthio, craig mullins."
#     ]

validation_prompts = [
    "a photograph of an astronaut riding a horse",
    "this bird has a red crown with black throat and red belly.",
    "A photo of a Corgi dog riding a bike in Times Square. It is wearing sunglasses and a beach hat.",
    "Cute and adorable ferret wizard, wearing coat and suit, steampunk, lantern, anthromorphic, Jean paptiste monge, playing the violin, oil painting",
    "Cluttered house in the woods, anime, oil painting, high resolution, cottagecore, ghibli inspired, 4k",
    "Funky pop Yoda figurine, made of plastic, product studio shot, on a white background, diffused lighting, centered",
    "A photo of a corgi dog wearing a wizard hat playing guitar on the top of a mountain.",
    "A panda making latte art.",
    "A black apple and a green backpack.",
    "A cute girl riding a horse",
    "Three spheres made of glass falling into ocean. Water is splashing. Sun is setting",
    "One cat and two dogs sitting on the grass",
    "A small blue book sitting on a large red book.",
    "A cozy living room with a painting of a corgi on the wall above a couch anda round coffee table in front of a couch and a vase of flowers on a coffee table",
    "New York Skyline with Hello World written with fireworks on the sky.",
    "A young beautiful girl in a translucent raincoat in colorful",
    "A photo of a confused grizzly bear in calculus class.",
    "A photo of a person with the head of a cow, wearing a tuxedo and black bowtie.  Beach wallpaper in the background.",
]

if validation_prompts is not None:
    pipeline = pipeline.to(device)
    pipeline.torch_dtype = weight_dtype
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_xformers_memory_efficient_attention()

    for i in range(len(validation_prompts)):
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        with torch.autocast("cuda"):
            image = pipeline(validation_prompts[i], num_inference_steps=20, generator=generator, num_images_per_prompt=1).images[0]
        image.save(f"img-{i:04f}.png")
        