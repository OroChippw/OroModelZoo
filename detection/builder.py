import torch.nn as nn
from utils.registry import Registry , build_from_cfg

MODELS = Registry("models")
BACKBONES = Registry("backbones")
LOSS = Registry("losses")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")

def build(cfg , registry , default_args = None): 
    return build_from_cfg(cfg , registry , default_args)

def build_model(cfg , preset_cfg , **kwargs):
    default_args = {
        'PRESET': preset_cfg
    }
    for key , value in kwargs.items():
        default_args[key] = value
    return build(cfg , MODELS , default_args)

def build_backbone(cfg , preset_cfg , **kwargs):
    default_args = {
        'PRESET': preset_cfg
    }
    for key , value in kwargs.items():
        default_args[key] = value
    return build(cfg , BACKBONES , default_args)

def build_loss(cfg):
    return build(cfg , LOSS)

def build_dataset(cfg , preset_cfg , **kwargs):
    default_args = {
        'PRESET': preset_cfg
    }
    for key , value in kwargs.items():
        default_args[key] = value
    return build(cfg , DATASETS , default_args)

def build_transforms(cfg):
    return build(cfg , TRANSFORMS)
        
