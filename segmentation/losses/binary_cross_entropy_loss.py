import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSS
# https://blog.csdn.net/code_plus/article/details/115739343

@LOSS.register_module()
class BCELoss(nn.Module):
    def __init__(self , loss_weight=1.0 , reduction='mean' , use_sigmoid=True) -> None:
        super(BCELoss , self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.use_sigmoid = use_sigmoid
        
    def forward(self , predict , target):
        if len(predict.shape) != len(target.shape):
            label = torch.unsqueeze(target , 1)
        if self.use_sigmoid:
            loss_ = F.binary_cross_entropy_with_logits(predict , target.float())
        else:
            loss_ = F.binary_cross_entropy(predict , target.float())
        return loss_