from utils import build_from_cfg ,Registry

BACKBONE = Registry('backbone')

def build_backbone(cfg):
    return build_from_cfg(cfg, BACKBONE)