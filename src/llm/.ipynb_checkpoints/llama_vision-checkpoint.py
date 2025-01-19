import torch
from PIL import Image
from huggingface_hub import login
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

def process_llm_output(output_text):
    pass

def get_llm_model(config):
    login(token=config.token)
    model = MllamaForConditionalGeneration.from_pretrained(config.model,torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(config.model)
    return model, processor

def generate_prompt(image_path, prompt, config):
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
    output = model.generate(**inputs, **config.model_kwargs)
    output_text = processor.decode(output[0])

    return output_text

if __name__ == '__main__':
    output_text = generate_prompt('workspace/InpaLLa/results/new_catdog/new_catdog.jpg', 'Change cat to dog', None)
    print(output_text)
    

    
    
    
    