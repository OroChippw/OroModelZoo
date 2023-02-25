import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSS

@LOSS.register_module()
class DiceLoss(nn.Module):
    def __init__(self , loss_weight=1.0 , beta=1 , smooth=1e-5) -> None:
        super(DiceLoss , self).__init__()
        self.loss_weight = loss_weight
        self.beta = beta
        self.smooth = smooth
        
    def forward(self , predict , target):
        if len(predict.shape) != len(target.shape):
            target = torch.unsqueeze(target , 1)
        score = ""
        dice_loss = 1 - torch.mean(score)
        loss_ = dice_loss
        return loss_