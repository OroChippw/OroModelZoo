from utils import build_from_cfg ,Registry

BACKBONE = Registry('backbone')
LOSSES = Registry('losses')

def build(cfg , registry , args = None):
    return build_from_cfg(cfg, registry , args)

def build_backbone(cfg):
    return build(cfg , BACKBONE)

def build_losses(cfg):
    return build(cfg , LOSSES)