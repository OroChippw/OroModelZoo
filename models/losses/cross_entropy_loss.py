import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES

def cross_entropy(pred , label , weight = None, reduction = 'mean',
                  avg_factor = None , class_weight = None):
    """
    Func:
        Calculate the CrossEntropy loss
    Args:
        pred(torch.Tensor):The prediction with shape (N, C), C is the number
            of classes.
        label(torch.Tensor):The gt label of the prediction.
        weight(torch.Tensor):Sample-wise loss weight.
        reduction(str):The method used to reduce the loss.
        avg_factor(int):Average factor that is used to average
            the loss. Defaults to None.
        class_weight(torch.Tensor):The weight for each class with
            shape (C), C is the number of classes. Default None.
    Returns:
        The calculated loss
    """
    loss = F.cross_entropy(pred , label , weight=class_weight , reduction='none')

    if weight is not None:
        weight = weight.float()
    loss = loss
    return loss