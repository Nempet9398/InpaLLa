from src.inpainting import *
from diffusers.utils import load_image
import os
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import hydra
import cv2
import numpy as np

def preprocess_mask(mask):
    mask = mask.convert('L')
    masked = mask.filter(ImageFilter.GaussianBlur(radius=30))
    masked = np.asarray(masked)
    masked =  np.where(masked > 10, 255, 0).astype(np.uint8)
    masked = Image.fromarray(masked)    
    return masked

def inpaint_image(model,image, mask, prompt, config): #guidance_scale,num_inference_steps):

    mask = preprocess_mask(mask)    
    ## 추후에 input으로 받아서 gradio와 연동시키기
    # guidance = guidance_scale
    # infer_step = num_inference_steps
    
    h, w = mask.height, mask.width
    
    gen_image = model(
        prompt= prompt,
        image=image,
        mask_image=mask,
        height=h,
        width=w,
        guidance_scale=30,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

    return gen_image

def get_inpaint():
    initialize(version_base='1.3', config_path='configs') 
    cfg = compose(config_name="base")

    model = get_inpainting_model(cfg.inpainting)
    model.to(cfg.device)

    return model, cfg

def inpaint(model, cfg:DictConfig):
    mask = load_image('/workspace/InpaLLa/results/Tobigs/Tobigs_mask_0.jpg')
    img = load_image('/workspace/InpaLLa/results/Tobigs/Tobigs.png')
    prompt = 'A Character holding axe'
    gen_image = inpaint_image(model, img ,mask, prompt, cfg.inpainting)
    gen_image.save(f'/workspace/InpaLLa/results/Tobigs/Tobigs_{prompt}.jpg')
    
if __name__=='__main__':
    model, cfg = get_inpaint()
    inpaint(model, cfg)