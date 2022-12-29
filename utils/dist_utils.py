import os

import torch
from torch import distributed as dist

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else :
        rank = 0
        world_size = 1
    return rank , world_size

def init_dist_pytorch(backend : 'str' , **kwargs):
    rank = int(os.environ['RANk'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend , **kwargs)