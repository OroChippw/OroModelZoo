import math
import torch.nn as nn
from .base_module import BaseModule
from .utils import InvertedresBlock

def _make_divisible(channels, divisor, min_channels=None):
    '''
    Functions:
        ensures that all layers have a channel number that is divisible by 8
    '''
    if min_channels is None:
        min_channels = divisor
    new_channels = max(min_channels,
                       int(channels + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels


# BaseConv or # Depthwise Separable Convolution，DSC
class Conv(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.Conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        result_ = self.Conv2d(x)
        return result_


class MobileNetv2(BaseModule):
    def __init__(self, widen_factor: float = 1.0, num_classes: int = 1000, pretrained=None):
        super(MobileNetv2, self).__init__()
        self.pretrained = pretrained

        # setting of inverted residual blocks
        # t (expand_ratio), c (output_channels), n (block replace num), s (first block stride)
        #
        self.cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        layers = []
        # building first layer
        input_channels = _make_divisible(32 * widen_factor,
                                         divisor=4 if widen_factor == 0.1 else 8)
        layers.append(Conv(in_channels=3, out_channels=input_channels, stride=2))

        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            out_channels = _make_divisible(c * widen_factor,
                                           divisor=4 if widen_factor == 0.1 else 8)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    InvertedresBlock(in_channels=input_channels, out_channels=out_channels,
                                     stride=stride, expand_ratio=t))
                input_channels = out_channels

        # building last several layers
        output_channels = _make_divisible(1280 * widen_factor,
                                          divisor=4 if widen_factor == 0.1 else 8)
        layers.append(Conv(in_channels=input_channels, out_channels=output_channels, stride=1))
        # building net skeleton
        self.skeleton_ = nn.Sequential(*layers)

        self.pool_ = nn.AdaptiveAvgPool2d((1, 1))

        self.head_ = nn.Linear(output_channels, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        input_ = x
        temp_ = self.skeleton_(input_)
        temp_ = self.pool_(temp_)
        temp_ = temp_.view(x.size(0), -1)
        result_ = self.head_(temp_)
        return result_
