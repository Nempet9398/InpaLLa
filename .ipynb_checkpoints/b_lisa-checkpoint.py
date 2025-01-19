from flask import Flask, request, jsonify
from src.mllm import get_mllm_model  # 실제 프로젝트 구조에 맞게 import 수정
import os
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import cv2
import numpy as np
from PIL import Image
import torch
import tempfile

app = Flask(__name__)

# -------------------------------------------------
# 1. 모델 로드 (서버 시작 시 한 번만 실행)
# -------------------------------------------------
def get_mllm():
    initialize(version_base='1.3', config_path='configs') 
    cfg = compose(config_name="base")

    model = get_mllm_model(cfg.mllm)
    model.to(cfg.device)

    return model, cfg

model, cfg = get_mllm()
print("LiSA 모델 로드 완료")

# -------------------------------------------------
# 2. Segmentation 함수 정의
# -------------------------------------------------
def seg_image(model, config, image_path, prompt):
    text_output, pred_masks, image_np = model.inference(image_path, prompt)
    return text_output, pred_masks, image_np

def save_mask(text_output, pred_masks, image_np, save_dir):
    mask_paths = []
    masked_image_paths = []

    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        # 마스크 이미지 저장
        # mask_filename = f"mask_{i}.jpg"
        mask_filename = "mask.jpg"

        save_mask_path = os.path.join(save_dir, mask_filename)
        cv2.imwrite(save_mask_path, pred_mask * 255)  # 0 or 255
        mask_paths.append(save_mask_path)
        print(f"{save_mask_path} has been saved.")

        # 마스크가 적용된 이미지 저장
        # save_imgnp_path = os.path.join(save_dir, f"masked_img_{i}.jpg")
        save_imgnp_path = os.path.join(save_dir, "masked_img.jpg")
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + (pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0])) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_imgnp_path, save_img)
        masked_image_paths.append(save_imgnp_path)
        print(f"{save_imgnp_path} has been saved.")

    return mask_paths, masked_image_paths

# -------------------------------------------------
# 3. Flask 라우트 정의
# -------------------------------------------------
@app.route('/lisa', methods=['POST'])
def perform_segmentation():
    try:
        # 1. 파일과 프롬프트 받기

        prompt = request.form.get('prompt', '')
        user_id = request.form.get('user_id')

 
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # 2. 임시 디렉토리에 이미지 저장
        # 사용자별 작업 디렉토리 설정
        user_dir = os.path.join('./data', user_id)
        os.makedirs(user_dir, exist_ok=True)  # 디렉토리 생성
        image_path = os.path.join(user_dir, f"{user_id}_image.jpg")  # 사용자 ID 포함 파일 이름

        # 3. Segmentation 수행
        text_output, pred_masks, image_np = seg_image(model, cfg, image_path, prompt)



            # 5. 마스크 및 마스크 적용 이미지 저장
        mask_paths, masked_image_paths = save_mask(text_output, pred_masks, image_np, user_dir)

            # 6. 결과 경로를 클라이언트에 반환
        return jsonify({
                "mask_paths": mask_paths,
                "masked_image_paths": masked_image_paths
            }), 200

    except Exception as e:
        print(f"Error during segmentation: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# 4. 서버 실행
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
