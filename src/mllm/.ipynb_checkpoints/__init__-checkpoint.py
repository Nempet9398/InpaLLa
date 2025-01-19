from .LISA.lisa import *

def get_mllm_model(config):
    mllm = LISA(config)
    return mllm