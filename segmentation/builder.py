import torch.nn as nn
from utils.registry import Registry , retrieve_from_cfg , build_from_cfg

MODELS = Registry("models")

def build(cfg , registry , default_args = None): 
    return build_from_cfg(cfg , registry , default_args)

def build_model(cfg , preset_cfg , **kwargs):
    default_args = {
        'PRESET': preset_cfg
    }
    for key , value in kwargs.items():
        default_args[key] = value
    return build(cfg , MODELS , default_args)
        
