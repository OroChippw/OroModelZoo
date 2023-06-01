import torch
import torch.nn as nn

from .base_module import BaseModule
from ..builder import MODELS

@MODELS.register_module()
class VisionTransformer(BaseModule):
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        result_ = x
        return result_
    
if __name__ == '__main__':
    model = VisionTransformer()
    