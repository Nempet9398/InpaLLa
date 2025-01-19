from src.mllm import *
import os
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import hydra
import cv2
import numpy as np

def seg_image(model, config):
    text_output, pred_masks, image_np = model.inference(config.mllm.image_path, config.mllm.prompt)
    return text_output, pred_masks, image_np

def save_mask(text_output ,pred_masks, image_np, save_path):
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0
        
        save_mask_path = "{}/{}_mask_{}.jpg".format(
            save_path, save_path.split("/")[-1], i
        )
        

        cv2.imwrite(save_mask_path, pred_mask * 100)
        print("{} has been saved.".format(save_path))
        
        save_imgnp_path = "{}/{}_masked_img_{}.jpg".format(
            save_path, save_path.split("/")[-1], i
        )
        
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_imgnp_path, save_img)
        print("{} has been saved.".format(save_path))

def get_mllm():
    initialize(version_base='1.3', config_path='configs') 
    cfg = compose(config_name="base")

    model = get_mllm_model(cfg.mllm)
    model.to(cfg.device)

    return model, cfg

def segment(model, cfg:DictConfig):
    # file_name, file_extension = os.path.splitext(os.path.basename('/workspace/InpaLLa/results/Tobigs/Tobigs.png'))
    cfg.mllm.image_path = '/workspace/InpaLLa/results/Tobigs/Tobigs.png' 
    #이걸 이용해서 segment해야 하기 때문에 이미지를 업로드하면 이미지를 path에 저장하도록 해야함 ## 원본 이미지 - 저장
    cfg.mllm.prompt = 'Segment what the character is holding.' # 프롬프트
    
    text_output, pred_masks, image_np = seg_image(model, cfg)

    save_path = '/workspace/InpaLLa/results/Tobigs' ## 마스크 저장 - 변경하기
    save_mask(text_output, pred_masks, image_np, save_path)

if __name__=='__main__':
    model, cfg = get_mllm()
    segment(model, cfg)
    
    