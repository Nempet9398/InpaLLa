from .inpainting import *
from .mllm import *
import os

class InpaLLA():
    def __init__(self,cfg):
        self.config = cfg
        self.inpainting = get_inpainting_model(self.config.inpainting)
        self.mllm = get_mllm_model(self.config.mllm)
        

    def seg_image(self, image_path:str, prompt:str):
        self.mllm.to(self.config.device)
        text_output, pred_masks, image_np = self.mllm.inference(image_path, prompt)
        return text_output, pred_masks, image_np

    def preprocess_mask(self, mask):
        mask = mask.convert('L')
        masked = mask.filter(ImageFilter.GaussianBlur(radius=30))
        masked = np.asarray(masked)
        masked =  np.where(masked > 10, 255, 0).astype(np.uint8)
        masked = Image.fromarray(masked)    
        return masked
        
    def inpaint_image(self, image, mask, prompt): #guidance_scale,num_inference_steps):
        
        '''
        mask = mask.convert('L')
        masked = mask.filter(ImageFilter.GaussianBlur(radius=30))
        masked = np.asarray(masked)
        masked =  np.where(masked > 10, 255, 0).astype(np.uint8)
        masked = Image.fromarray(masked)
        '''    
        ## 추후에 input으로 받아서 gradio와 연동시키기
        # guidance = guidance_scale
        # infer_step = num_inference_steps
        self.inpainting.to(self.config.device)
        
        h, w = mask.height, mask.width
        
        image = self.inpainting(
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
    