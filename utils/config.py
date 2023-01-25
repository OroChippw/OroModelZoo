import yaml
from easydict import EasyDict as edict

def update_config(cfg):
    with open(cfg) as f:
        config_ = edict(yaml.load(f,Loader=yaml.FullLoader))
        return config_