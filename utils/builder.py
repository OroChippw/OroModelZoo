import torch.nn as nn

from .registry import Registry , retrieve_from_cfg , build_from_cfg

LOSS = Registry("loss")
DATASET = Registry("loss")

