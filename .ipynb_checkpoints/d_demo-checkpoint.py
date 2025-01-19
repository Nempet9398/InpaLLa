import requests
import gradio as gr
import os
import tempfile
import uuid
from diffusers.utils import load_image
from PIL import Image

# 사용자 ID 생성 함수
def generate_user_id():
    return str(uuid.uuid4())


MAX_SIZE = (1024, 1024)  # 최대 크기 (너비, 높이)

# 이미지 크기 제한 함수
def save_image_with_max_size(image, save_path, max_size):
    """
    이미지를 지정된 최대 크기로 조정하여 저장합니다.
    - image: PIL.Image 객체
    - save_path: 저장할 파일 경로
    - max_size: (width, height) 형태의 최대 크기
    """
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        # 크기 제한 적용
        image.thumbnail(max_size, Image.Resampling.LANCZOS)  # 비율 유지하며 크기 조정
    image.save(save_path, format="JPEG")  # JPEG 형식으로 저장
    print(f"Image saved at {save_path} with size {image.size}")

def pipeline(input_image_path, prompt, denoising_strength, num_inference_steps, guidance_scale, user_id):
    print(2)

    try:
        print(3)
        # -> 외부 포트 연결해서 받아오기 !! 5003 넣어서 나온 값 넣어주기
        # 1. LLM 호출: Segmentation Prompt와 Inpainting Prompt 생성
        # llm_url = 'http://localhost:5003/llm'  # LLM API URL
        llm_url ='http://localhost:5004/llm' 
        llm_response = requests.post(
            llm_url,
            data={'prompt': prompt, 'user_id': user_id}
        )
        print(4)
        ## LLM 특징 prompt 생성
        # image 원본을 받아서 임시경로에 저장
        # 임시경로에 저장된 이미지와 prompt를 통해 output_text 생성
        # ianpainting, segmentation 두개를 json으로 받음

        if llm_response.status_code != 200:
            return f"Error in LLM API: {llm_response.json().get('error', 'Unknown error')}"

        llm_result = llm_response.json()
        inpainting_prompt = llm_result['inpainting_prompt']
        segmentation_prompt = llm_result['segmentation_prompt']
        print(inpainting_prompt)
        print(segmentation_prompt)
        print(5)
        # 2. Segmentation 호출: 마스크 생성
        lisa_url = 'http://localhost:5001/lisa'  # Segmentation API URL

        segmentation_response = requests.post(
                lisa_url,
                data={'prompt': segmentation_prompt, 'user_id':user_id}
            )

        ## LISA 특징
        # 이미지 원본 받아서 image 임시경로로 저장
        # image 경로 저장해서 이 경로로 진행
        # mask도 임시 경로에 저장
        # 진행
        
        if segmentation_response.status_code != 200:
            return f"Error in Segmentation API: {segmentation_response.json().get('error', 'Unknown error')}"

        segmentation_result = segmentation_response.json()
        mask_paths = segmentation_result.get('mask_paths', [])
        print(6)
        if not mask_paths:
            return "No mask paths returned by segmentation API."

        mask_path = mask_paths[0]  # 첫 번째 마스크 사용

        # 3. Inpainting 호출: 최종 이미지 생성
        flux_url = 'http://213.173.105.10:29012/flux'  # Inpainting API URL

        inpainting_response = requests.post(
            flux_url,
            data={
                'prompt': inpainting_prompt,
                'guidance_scale': guidance_scale,
                'step': num_inference_steps,
                'user_id': user_id
            }
        )

        if inpainting_response.status_code != 200:
            return f"Error in Inpainting API: {inpainting_response.json().get('error', 'Unknown error')}"
        print(7)
        ## Flux 특징
        # 
        inapainting_result = inpainting_response.json()
        final_path = inpainting_result.result_path

        # 4. 최종 이미지 반환

        return final_path

    except Exception as e:
        return f"Pipeline error: {str(e)}"

# Gradio 인터페이스
def process_image(input_image, prompt, denoising_strength, num_inference_steps, guidance_scale,user_id):
    if input_image is None:
        return "Error: Input image is required."

    user_id = generate_user_id()
    print(user_id,'User-ID')
    # 사용자 ID 및 디렉토리 설정
    user_dir = os.path.join("./data", user_id)
    os.makedirs(user_dir, exist_ok=True)  # 디렉토리 생성
    
    # Gradio에서 전달된 PIL 이미지를 저장
    input_filename = f"{user_id}_image.jpg"
    input_image_path = os.path.join(user_dir, input_filename)
    
    # 크기 제한을 적용하여 저장
    save_image_with_max_size(input_image, input_image_path, (1024,1024))

    print(1)
    # 파이프라인 실행
    result = pipeline(input_image_path, prompt, denoising_strength, num_inference_steps, guidance_scale, user_id)

    final_path = f"./data/{user_id}/inpainted_image.jpg"
    final_mask_path = f"./data/{user_id}/masked_img.jpg"
    return final_path, final_mask_path  # 최종 이미지 경로 반환

# Gradio UI 정의
with gr.Blocks() as demo:
    gr.Markdown("## InpaLLa : Inapainting with mLLM Archtecture.")


    
    # 사용자 ID 생성


    # 입력 섹션
    with gr.Row():
        input_image = gr.Image(type="pil", label="입력 이미지")
        input_prompt = gr.Textbox(
            label="프롬프트", 
            placeholder="변환할 이미지와 세부 사항을 적어주세요 (예: 고양이를 강아지로 바꿔주세요)"
        )

    # 파라미터 섹션
    with gr.Row():
        denoising_strength = gr.Slider(
            0.0, 0.5, value=0.5, step=0.01, 
            label="Denoising Strength", 
            info="낮을수록 원본 이미지와 유사하고, 높을수록 창의적인 결과가 나옵니다."
        )
        num_inference_steps = gr.Slider(
            30, 50, value=50, step=1, 
            label="num_inference_steps", 
            info="많을수록 더 좋은 품질을 생성하지만 처리 시간이 늘어납니다."
        )
        guidance_scale = gr.Slider(
            10.0, 50.0, value=25, step=1, 
            label="guidance_scale", 
            info="높을수록 프롬프트 반영도가 증가됩니다."
        )

    # 출력 섹션
    with gr.Row():
        output_image = gr.Image(type="filepath", label="결과 이미지")
        mask_image = gr.Image(type="filepath", label="마스크 이미지")  # 추가

    # 실행 버튼
    submit_button = gr.Button("Inapinting")
    
    # 사용자 ID를 함께 전달
    submit_button.click(
        process_image,
        inputs=[input_image, input_prompt, denoising_strength, num_inference_steps, guidance_scale],
        outputs=[output_image, mask_image]
    )

if __name__ == "__main__":
    demo.launch(share=True)
