import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSS

class FocalLoss():
    def __init__(self) -> None:
        pass
    
    def forward(self , predict , target):
        return ""