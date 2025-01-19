import torch

# PyTorch 버전 확인
print(f"PyTorch Version: {torch.__version__}")

# CUDA 사용 가능 여부 확인
print(f"Is CUDA available: {torch.cuda.is_available()}")

# CUDA 버전 확인 (사용 가능할 때만 출력됨)
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


from transformers import TRANSFORMERS_CACHE

print("현재 저장 경로:", TRANSFORMERS_CACHE)
