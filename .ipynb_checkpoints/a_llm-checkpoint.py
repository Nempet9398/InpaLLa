from flask import Flask, request, jsonify
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import os
import tempfile
from src.llm import *
from diffusers.utils import load_image
import re
import hydra
import cv2
import numpy as np

app = Flask(__name__)

# -------------------------------------------------
# 1. 모델 로드 (서버 시작 시 한 번만 실행)
# -------------------------------------------------
def get_llm():
    initialize(version_base='1.3', config_path='configs') 
    cfg = compose(config_name="base")

    model, llm_processor = get_llm_model(cfg.llm)
    model.to(cfg.device)

    return model, llm_processor, cfg

model, llm_processor, cfg = get_llm()
print("LLM 모델 로드 완료")

# -------------------------------------------------
# 2. Helper 함수 정의
# -------------------------------------------------
def process_prompt(output_text):
    """
    LLM 출력 텍스트를 파싱하여 Inpainting과 Segmentation 프롬프트로 나눕니다.
    """
    prompts = output_text.split("\n")  # Split by newlines
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

def generate_prompt(model, processor, image_path, prompt, config):
    """
    LLM을 통해 Inpainting과 Segmentation 프롬프트를 생성합니다.
    """
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
    b, l = inputs['input_ids'].size()
    output = model.generate(**inputs, max_new_tokens=512)
    output = output[:, l:]
    output_text = processor.decode(output[0], skip_special_tokens=True)

    return output_text

# -------------------------------------------------
# 3. Flask 라우트 정의
# -------------------------------------------------
@app.route('/llm', methods=['POST'])
def generate():
    try:
        # 1. 이미지와 프롬프트 입력 받기

        if 'prompt' not in request.form:
            return jsonify({"error": "Prompt is required"}), 400

        # 파일 처리

        prompt = request.form.get('prompt')
        user_id = request.form.get('user_id')

        ## 유저 specific path 생성
        user_dir = os.path.join("./data", user_id)
        os.makedirs(user_dir, exist_ok=True)  # 디렉토리 생성

        # 이미지 저장 경로 설정
        # 사용자별 이미지 저장 경로 설정
        image_filename = f"{user_id}_image.jpg"  # 파일 이름 생성
        image_path = os.path.join(user_dir, image_filename)  # 전체 경로
        
        # 프롬프트 생성
        output_text = generate_prompt(model, llm_processor, image_path, prompt, cfg)

        # 텍스트 처리
        inpainting, segmentation = process_prompt(output_text)

        # 결과 반환
        return jsonify({
            "inpainting_prompt": inpainting,
            "segmentation_prompt": segmentation
        }), 200


    except Exception as e:
        print(f"Error during LLM processing: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------
# 4. 서버 실행
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
