import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, \
    DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np
from ip_adapter import StoryAdapterXL
from ip_adapter import IPAdapter
import os
import json
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_model_path', default=r"ckpt/story-ad/RealVisXL_V4", type=str)
parser.add_argument('--image_encoder_path', type=str, default=r"ckpt/ipa/sdxl_models/image_encoder")
parser.add_argument('--ip_ckpt', default=r"ckpt/ip-adapter_sdxl.bin", type=str)
parser.add_argument('--style', type=str, default='storybook', choices=["comic","film","realistic"])
parser.add_argument('--device', default="cuda", type=str)

args = parser.parse_args()

base_model_path = args.base_model_path
image_encoder_path = args.image_encoder_path
ip_ckpt = args.ip_ckpt
device = args.device
style = args.style

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

# load SD pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    feature_extractor=None,
    safety_checker=None
)

# load story-adapter
storyadapter = StoryAdapterXL(pipe, image_encoder_path, ip_ckpt, device)
out_dir = 'subtitles_story'
os.makedirs(out_dir, exist_ok=True)
seed = 42
def apply_character_prompts(description: str, character_dict: dict) -> str:
    """
    替换描述中的角色名为他们的完整 prompt
    """
    for name, char in character_dict.items():
        if name in description:
            description = description.replace(name, char['prompt'])
    return description


with open("story_json/story_1.json", 'r', encoding='utf-8') as f:
    story_data = json.load(f)

    character_prompts = story_data["Characters"]
    shot_descriptions = story_data["Shots"]


    styles = "storybook"
    prompts = [apply_character_prompts(shot["Static Shot Description"], character_prompts) for shot in shot_descriptions]

    os.makedirs(f'./story_test/story_1_{seed}', exist_ok=True)
    os.makedirs(f'./story_test/story_1_{seed}/results_xl', exist_ok=True)

    # 第一次生成
    for i, text in enumerate(prompts):
        print(f"[{seed}] prompt {i+1}: {text}")
        images = storyadapter.generate(pil_image=None, num_samples=1, num_inference_steps=50, seed=seed,
                                       prompt=text, scale=0.3, use_image=False, style=styles)
        grid = image_grid(images, 1, 1)
        grid.save(f'./story_test/story_1_{seed}/results_xl/img_{i}.png')

    # 后续 scale 细化生成
    images = []
    for y in range(len(prompts)):
        image = Image.open(f'./story_test/story_1_{seed}/results_xl/img_{y}.png')
        image = image.resize((256, 256))
        images.append(image)

    scales = np.linspace(0.3, 0.5, 5)
    for i, scale in enumerate(scales):
        new_images = []
        metadata = []
        os.makedirs(f'./story_test/story_1_{seed}/results_xl{i+1}', exist_ok=True)
        for y, text in enumerate(prompts):
            print(f"Epoch {i+1}, image {y}")
            image = storyadapter.generate(pil_image=images, num_samples=1, num_inference_steps=50, seed=seed,
                                          prompt=text, scale=scale, use_image=True, style=styles)
            new_images.append(image[0].resize((256, 256)))
            grid = image_grid(image, 1, 1)
            save_path = f'./story_test/story_1_{seed}/results_xl{i+1}/img_{y}.png'
            if int(i) == 4:
                metadata.append({
                    "scene_number": y+1,
                    "prompt": text,
                    "image_path": save_path
                })
            grid.save(save_path)
        images = new_images

    with open(os.path.join(out_dir, f"story_1_{seed}.json"), "w") as f:
        json.dump(metadata, f, indent=2)
