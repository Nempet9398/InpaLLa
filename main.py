from src import InpaLLA
from omegaconf import DictConfig, OmegaConf
import hydra
import cv2
import numpy as np

def save_mask(text_output ,pred_masks, image_np):
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0
        '''
        save_path = "{}/{}_mask_{}.jpg".format(
            self.config.save_path, image_path.split("/")[-1].split(".")[0], i
        )
        '''
        save_path = '/workspace/InpaLLa/results/test.png'
        cv2.imwrite(save_path, pred_mask * 100)
        print("{} has been saved.".format(save_path))
        '''
        save_path = "{}/{}_masked_img_{}.jpg".format(
            args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
        )
        '''
        save_path = '/workspace/InpaLLa/results/test1.png'
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)
        print("{} has been saved.".format(save_path))

@hydra.main(config_path='configs', config_name="base")
def main(cfg:DictConfig):
    model = InpaLLA(cfg)
    text_output, pred_masks, image_np = model.seg_image('/workspace/InpaLLa/results/new_catdog.jpg', 'Segment cat.')
    save_mask(text_output, pred_masks, image_np)
    

if __name__ == '__main__':
    main()