import yaml
from easydict import EasyDict as edict

def update_config(cfg):
    with open(cfg) as f:
        config = edict(yaml.load(f,Loader=yaml.FullLoader))
        return config