import torch.nn as nn
from models.backbones import BaseModule

# BaseConv or # Depthwise Separable Convolution，DSC
class Conv(BaseModule):
    def __init__(self , in_channels , out_channels , kernel_size = 3 , 
                stride :int = 1 , padding :int = 1 , groups : int = 1):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.Conv2d = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
            kernel_size = kernel_size , stride = stride , padding = padding , 
            groups = groups , bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self , x) : 
        result_ = self.Conv2d(x)
        return result_

class InvertedresBlock(BaseModule):
    '''
    InvertedResidual block for MobileNetV2
    先升维再降维，降维的时候用的线性激活函数，防止信息丢失，
    shortcut连接的是低维部分通道数较少的tensor，stride为1时才有残差结构
    ReLU6在低精度浮点数数上有比较好的表示性能
    expressiveness 非线性变换能力
    Args:
        expand_ratio : expansion factor升维的倍数

    '''
    def __init__(self , in_channels , out_channels , stride , expand_ratio):
        super(InvertedresBlock, self).__init__()

        hidden_channel = round(in_channels * expand_ratio)
        self.use_shortcut = (stride == 1) and (in_channels == out_channels)
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
            f'But received {stride}.'

        layers = []
        if expand_ratio != 1 : 
            # Ascension
            layers.append(Conv(in_channels, hidden_channel , kernel_size=1))
        
        layers.extend([
            # Depthwise Conv
            Conv(hidden_channel, hidden_channel ,stride=stride , 
                groups=hidden_channel),
            # Linear Bottleneck : Pointwise Conv ,  is equivalent to a Linear 
            nn.Conv2d(hidden_channel, out_channels, kernel_size = 1 , bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.Inverted_res = nn.Sequential(*layers)

    def forward(self , x):
        input_ = x
        if self.use_shortcut : 
            return input_ + self.Inverted_res(input_)
        else :
            return self.Inverted_res(input_)
    