from .flux.flux import *

def get_inpainting_model(config):
    pipe = get_model(config)
    return pipe

