import torch.nn as nn

from .registry import Registry , retrieve_from_cfg , build_from_cfg

MODELS = Registry("models")
LOSS = Registry("loss")
DATASETS = Registry("datasets")

def build(cfg , registry , default_args = None):
    if isinstance(cfg , list):
        modules = [
            build_from_cfg(cfg_ , registry , default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else :
        return build_from_cfg(cfg , registry , default_args)

def build_model(cfg , preset_cfg , **kwargs):
    default_args = {
        'PRESET': preset_cfg
    }
    for key , value in kwargs.items():
        default_args[key] = value
    return build(cfg , MODELS , default_args)
        
def build_loss(cfg):
    return build(cfg , LOSS)

def build_dataset(cfg , preset_cfg , **kwargs):
    exec(f"from .segmentation.datasets import {cfg.type}")
    default_args = {
        'PRESET': preset_cfg
    }
    for key , value in kwargs.items():
        default_args[key] = value
    return build(cfg , DATASETS , default_args)

def retrieve_dataset(cfg):
    exec(f"from .segmentation.dataset import {cfg.type}")
    return retrieve_from_cfg(cfg , DATASETS)