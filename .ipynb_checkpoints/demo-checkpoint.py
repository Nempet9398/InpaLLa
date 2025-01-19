import gradio as gr
from src import InpaLLA
from omegaconf import DictConfig, OmegaConf
import hydra
from PIL import ImageOps

# ===== 1. InpaLLA 초기화 =====
# @hydra.main(config_path="configs", config_name="base")
# def load_model(cfg: DictConfig):
#     return InpaLLA(cfg)

# # 모델 초기화
# model = load_model()

# ===== 2. sLM 파트 함수 =====

## 바꿔놓아야함 (pending)

def prompt_to_model(prompt):
    """
    Prompt를 기반으로 MLLM 및 FLUX 용 프롬프트 생성
    """
    # 지금은 임시로 이거
    prompt_MLLM = f"MLLM-{prompt}"
    prompt_FLUX = f"FLUX-{prompt}"
    return prompt_MLLM, prompt_FLUX

# ===== 3. Gradio 작업 함수 =====
def process_image(image, prompt, denoising_strength, num_inference_steps, guidance_scale):
    """
    전체 변환 과정을 처리하는 함수
    """
    # # Prompt 변환
    # prompt_MLLM, prompt_FLUX = prompt_to_model(prompt)
    
    # # Segmentation 작업
    # text_output, pred_masks, image_np = model.seg_image(image_path=image, prompt=prompt_MLLM)
    
    # # Mask 전처리
    # processed_mask = model.preprocess_mask(pred_masks)
    
    # # Inpainting 작업
    # final_image = model.inpaint_image(
    #     image=image_np,
    #     mask=processed_mask,
    #     prompt=prompt_FLUX,
    #     denoising_strength=denoising_strength,
    #     num_inference_steps=num_inference_steps,
    #     guidance_scale=guidance_scale,
    # )
    
    # return final_image  # 최종 변환된 이미지를 반환
    # 파일 경로일 경우 이미지 열기
    if isinstance(image, str):
        image = Image.open(image)

    # 흑백 변환 작업
    final_image = ImageOps.grayscale(image)
    
    return final_image  # 최종 변환된 이미지를 반환

# ===== 4. Gradio 인터페이스 설정 =====

with gr.Blocks() as demo:
    # 제목
    gr.Markdown("## Unified Inpainting Model")

    # 입력 섹션
    with gr.Row():
        input_image = gr.Image(type="filepath", label="Input Image")
        
        input_prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt")

    # Hyperparameter 슬라이더
    with gr.Row():
        denoising_strength = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Denoising Strength")
        num_inference_steps = gr.Slider(1, 100, value=50, step=1, label="Number of Inference Steps")
        guidance_scale = gr.Slider(1.0, 20.0, value=7.5, step=0.1, label="Guidance Scale")

    # 출력 섹션
    with gr.Row():
        output_image = gr.Image(label="Output Image")

    # 버튼 섹션
    with gr.Row():
        submit_button = gr.Button("Request")
        reset_button = gr.Button("Reset")
        update_button = gr.Button("Update Prompt")

    # 버튼 동작 설정
    submit_button.click(
        process_image,
        inputs=[input_image, input_prompt, denoising_strength, num_inference_steps, guidance_scale],
        outputs=[output_image]
    )

    def reset_all():
        """
        모든 입력 필드 초기화
        """
        return None, "", 0.5, 50, 7.5, None

    reset_button.click(
        reset_all,
        inputs=[],
        outputs=[input_image, input_prompt, denoising_strength, num_inference_steps, guidance_scale, output_image]
    )

    update_button.click(
        process_image,
        inputs=[input_image, input_prompt, denoising_strength, num_inference_steps, guidance_scale],
        outputs=[output_image]
    )

# ===== 5. 애플리케이션 실행 =====
demo.launch(share=True)



## prompt - > [llm] -> (lisa_prompt, flux_prompt) -> lisa, flux -> inpaint
