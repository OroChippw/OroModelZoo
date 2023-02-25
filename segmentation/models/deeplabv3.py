import torch
import torch.nn as nn
from torchsummary import summary
from .base_module import BaseModule
from ..builder import MODELS

@MODELS.register_module()
class DeepLabV3p(BaseModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return super().forward(x)