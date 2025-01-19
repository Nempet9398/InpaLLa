from src.llm import *
from diffusers.utils import load_image
import os
import re
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import hydra
import cv2
import numpy as np

def process_prompt(output_text):
    prompts = output_text.split("\n")# Split by double newline
    prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
    
    prompt1 = prompts[0].split(':')[1].strip()
    prompt1_index = prompt1.find(",")
    if prompt1_index != -1:
        prompt1 = prompt1[prompt1_index+1:].strip()

    prompt2 = prompts[1].split(':')[1].strip()
    match = re.search(r'\[(.*?)\]', prompt2)
    if match:
        prompt2 = match.group(1)
    prompt2 = 'Segment ' + prompt2        
    return prompt1.strip(), prompt2.strip()

def get_llm():
    initialize(version_base='1.3', config_path='configs') 
    cfg = compose(config_name="base")

    model,llm_processor = get_llm_model(cfg.llm)
    model.to(cfg.device)

    return model, llm_processor ,cfg

def generate_prompt(model, processor, image_path, prompt, config):
    image = Image.open(image_path)
    user_prompt = prompt

    messages = [
    {"role": "system", "content":[
        {"type": "text", "text": '''You are a language model that generates prompts for inpainting and segmentation task. User provide two instruction and you must follow user's instruction.
output format must
Prompt1: user prompt1
Prompt2: user prompt2'''}
    ]},
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"Prompt1: What will this image be like if {user_prompt}?. Describe the image in three sentence. The response must be clear, not vague or abstract.\nPrompt2: Which object should be segmented in this image for {user_prompt}?. output format must be [Segment object]"}
    ]}
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    b,l = inputs['input_ids'].size()
    output = model.generate(**inputs, max_new_tokens=512)
    output = output[:,l:]
    output_text = processor.decode(output[0],skip_special_tokens=True)

    return output_text

if __name__=='__main__':

    model,llm_processor, cfg = get_llm()
    image_path = '/workspace/InpaLLa/results/Tobigs/Tobigs.png'
    prompt = 'Change bear character to cat character.'
    output_text = generate_prompt(model, llm_processor, image_path, prompt, cfg)

    inpainting, segmentation = process_prompt(output_text)
    print(inpainting)
    print(segmentation)