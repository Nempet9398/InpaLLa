import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from huggingface_hub import login
import numpy as np
from PIL import Image,ImageFilter

def get_model(config):
    
    login(token=config.token)
    pipe = FluxFillPipeline.from_pretrained(config.model, torch_dtype=torch.bfloat16, cache_dir=config.cache_dir)
    return pipe

def inpaint(model, image, mask, prompt):
    
    mask = mask.convert('L')
    masked = mask.filter(ImageFilter.GaussianBlur(radius=30))
    masked = np.asarray(masked)
    masked =  np.where(masked > 10, 255, 0).astype(np.uint8)
    masked = Image.fromarray(masked)

    h, w = mask.height, mask.width
    
    image = model(
        prompt= prompt,
        image=image,
        mask_image=masked,
        height=h,
        width=w,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

    return image