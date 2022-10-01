import torch
import torch.nn as nn
from ..base_module import BaseModule

# BaseConv or # Depthwise Separable Convolution，DSC
class Conv(BaseModule):
    def __init__(self , in_channels , out_channels , kernel_size = 3 , 
                stride :int = 1 , padding :int = 1 , groups : int = 1):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.Conv2d = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
            kernel_size = kernel_size , stride = stride , padding = padding , 
            group = 1 , bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self , x) : 
        result_ = self.Conv2d(x)
        return result_

class InvertedresBlock(BaseModule):
    '''
    InvertedResidual block for MobileNetV2
    Args:
        expand_ratio : expansion factor

    '''
    def __init__(self , in_channels , out_channels , stride , expand_ratio):
        super(InvertedresBlock, self).__init__()

        hidden_channel = round(in_channels * expand_ratio)
        self.use_shortcut = (stride == 1) and (in_channels == out_channels)
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'

        layers = []
        if expand_ratio != 1 : 
            layers.append(Conv(in_channels, hidden_channel , kernel_size=1))
        
        layers.append([
            # Depthwise Conv
            Conv(hidden_channel, hidden_channel ,stride=stride , 
                groups=hidden_channel),
            # Pointwise Conv ,  is equivalent to a Linear
            nn.Conv2d(hidden_channel, out_channels, kernel_size = 1 , bias=True),
            nn.BatchNorm2d(out_channels)
        ])

        self.Conv2d = nn.Sequential(*layers)

    def forward(self , x):
        input_ = x
        if self.use_shortcut : 
            return input_ + self.Conv2d(input_)
        else :
            return self.Conv2d(input_)
    