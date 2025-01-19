from flask import Flask, request, jsonify, send_file
from src.inpainting import *  # 실제 프로젝트 구조에 맞게 import 수정
from diffusers.utils import load_image
import os
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
import tempfile

app = Flask(__name__)

# -------------------------------------------------
# 1. 모델 로드 (서버 시작 시 한 번만 실행)
# -------------------------------------------------
def get_inpaint():
    initialize(version_base='1.3', config_path='configs') 
    cfg = compose(config_name="base")

    model = get_inpainting_model(cfg.inpainting)
    model.to(cfg.device)

    return model, cfg

model, cfg = get_inpaint()
print("Inpainting 모델 로드 완료")

# -------------------------------------------------
# 2. Inpainting 함수 정의
# -------------------------------------------------
def preprocess_mask(mask: Image.Image) -> Image.Image:
    """
    마스크 이미지에 블러를 주고,
    Threshold를 통해 깔끔한 마스크로 만들어주는 간단 전처리
    """
    mask = mask.convert('L')
    masked = mask.filter(ImageFilter.GaussianBlur(radius=30))
    masked = np.asarray(masked)
    masked = np.where(masked > 10, 255, 0).astype(np.uint8)
    masked = Image.fromarray(masked)
    return masked

def inpaint_image(model, image: Image.Image, mask: Image.Image, prompt: str, config: DictConfig, step, guidance) -> Image.Image:
    """
    Inpainting을 수행하는 함수
    - model: Inpainting model
    - image: 원본 이미지 (PIL)
    - mask: 마스크 이미지 (PIL)
    - prompt: 사용자 제공 프롬프트
    - config: Hydra로부터 가져온 inpainting 관련 설정
    """
    mask = preprocess_mask(mask)

    h, w = mask.height, mask.width

    gen_image = model(
        prompt=prompt,
        image=image,
        mask_image=mask,
        height=h,
        width=w,
        guidance_scale=guidance,
        num_inference_steps=step,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]

    return gen_image

# -------------------------------------------------
# 3. Flask 라우트 정의
# -------------------------------------------------
@app.route('/flux', methods=['POST'])
def perform_inpainting():
    try:
        # 1. 파일과 프롬프트 받기



        prompt = request.form.get('prompt', '')
        guidance_scale = request.form.get('guidance_scale', None)
        step = request.form.get('step', None)
        user_id = request.form.get('user_id')
        

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        if guidance_scale is None or step is None:
            return jsonify({"error": "Guidance scale and step are required"}), 400

        try:
            guidance_scale = float(guidance_scale)
            step = int(step)
        except ValueError:
            return jsonify({"error": "Invalid guidance_scale or step value"}), 400


        user_dir = os.path.join('./data', user_id)
        os.makedirs(user_dir, exist_ok=True)  # 디렉토리 생성

        
        # 2. 마스크 로드
        mask_path = os.path.join(user_dir,'mask.jpg')
        mask = load_image(mask_path)
        
        # 3. 이미지 로드
        image_filename = f"{user_id}_image.jpg"
        image_path = os.path.join(user_dir, image_filename)
        img = load_image(image_path)  # load_image가 PIL Image를 반환한다고 가정
                                  
        
        # 5. Inpainting 수행
        gen_image = inpaint_image(model, img, mask, prompt, cfg.inpainting, step, guidance_scale)

        # 6. 결과 이미지 저장
        result_filename = f"inpainted_image.jpg"
        result_path = os.path.join(user_dir, result_filename)
        gen_image.save(result_path)

        # 7. 결과 이미지 파일을 클라이언트에 반환
                # 7. 결과 이미지 경로를 클라이언트에 반환
        return jsonify({"result_path": result_path}), 200

    
        # # 2. 임시 디렉토리에 이미지 저장
        # with tempfile.TemporaryDirectory() as tmpdirname:
        #     original_image_path = os.path.join(tmpdirname, original_image_file.filename)
        #     mask_image_path = os.path.join(tmpdirname, mask_image_file.filename)
        #     original_image_file.save(original_image_path)
        #     mask_image_file.save(mask_image_path)

        #     # 3. 이미지 로드
        #     original_img = load_image(original_image_path)  # load_image가 PIL Image를 반환한다고 가정
        #     mask_img = load_image(mask_image_path)


        #     # 5. Inpainting 수행
        #     gen_image = inpaint_image(model, original_img, mask_img, prompt, cfg.inpainting, step, guidance_scale)

        #     # 6. 결과 이미지 저장
        #     result_filename = f"inpainted_{os.path.splitext(original_image_file.filename)[0]}.jpg"
        #     # result_path = os.path.join(tmpdirname, result_filename)
        #     # gen_image.save(result_path)
        #     gen_image.save(result_filename)
        #     # 7. 결과 이미지 파일을 클라이언트에 반환
        #     return send_file(result_path, mimetype='image/jpeg', as_attachment=True, attachment_filename=result_filename)

    except Exception as e:
        print(f"Error during inpainting: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# 4. 서버 실행
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
