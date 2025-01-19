from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
import torch
import argparse
from model.LISA import LISAForCausalLM
import cv2
import numpy as np
import torch.nn.functional as F
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def default_args():
    args = argparse.ArgumentParser()
    args.version = "xinlai/LISA-13B-llama2-v1"
    args.model_max_length = 512
    args.precision = "fp16"
    args.load_in_4bit = True
    args.load_in_8bit = False
    args.vision_tower = "openai/clip-vit-large-patch14"
    return args

class LISA:
    def __init__(self):
        self.args = default_args()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.version,
            cache_dir=None,
            model_max_length=self.args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.args.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

        torch_dtype = torch.float32
        if self.args.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif self.args.precision == "fp16":
            torch_dtype = torch.half

        kwargs = {"torch_dtype": torch_dtype}
        if self.args.load_in_4bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "load_in_4bit": True,
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_skip_modules=["visual_model"],
                    ),
                }
            )
        elif self.args.load_in_8bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_8bit=True,
                    ),
                }
            )

        self.model = LISAForCausalLM.from_pretrained(
            self.args.version, low_cpu_mem_usage=True, vision_tower=self.args.vision_tower, seg_token_idx=self.args.seg_token_idx, **kwargs
        )

        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(1024)  # 기본 이미지 크기 1024로 설정
        self.model.eval()
    
    def preprocess(self, x, 
                  pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
                  pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
                  img_size=1024):
        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def inference(self, image_path, prompt, use_mm_start_end=True):
        # 프롬프트 전처리
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if use_mm_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        # 이미지 로드 및 전처리
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        # CLIP 이미지 처리
        image_clip = self.clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
        if self.args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif self.args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        # SAM 이미지 처리
        image = self.transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda()
        
        if self.args.precision == "bf16":
            image = image.bfloat16()
        elif self.args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        # 토크나이저 처리
        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # 모델 추론
        output_ids, pred_masks = self.model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=self.tokenizer,
        )
        
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")

        return text_output, pred_masks, image_np
    

if __name__ == "__main__":
    lisa = LISA()
    text_output, pred_masks, image_np = lisa.inference("./imgs/blackpink.jpg", "Please segment Lisa in this figure.")
    import ipdb; ipdb.set_trace()