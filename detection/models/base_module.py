import torch.nn as nn

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def forward(self, x):
        pass

    def train(self, mode=True):
        pass