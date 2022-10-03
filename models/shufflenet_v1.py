import torch.nn as nn
from .base_module import BaseModule

class ShuffleNetv1(BaseModule):
    def __init__(self):
        super(ShuffleNetv1, self).__init__()

        self.skeleton_ = ""

        self.head_ = ""

    def forward(self, x):
        input_ = x
        temp_ = input_
        result_ = temp_
        return result_