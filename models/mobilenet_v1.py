import torch.nn as nn
import torch.nn.functional as F
from .base_module import BaseModule


# BaseConv
class Conv(BaseModule):
    def __init__(self, in_channels, out_channels,
                 stride: int = 1, padding: int = 1):
        super(Conv, self).__init__()
        self.Conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        result_ = self.Conv2d(x)

        return result_


# Depthwise Separable Convolution，DSC
class DSConvBlock(BaseModule):
    def __init__(self, in_channels, out_channels,
                 stride: int = 1, padding: int = 1):
        super(DSConvBlock, self).__init__()
        # Depthwise
        self.DepthwiseBlock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        # Pointwise
        self.PointwiseBlock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        input_ = x
        temp_ = F.relu(self.DepthwiseBlock(input_))
        result_ = F.relu(self.PointwiseBlock(temp_))

        return result_


class MobileNetv1(BaseModule):
    def __init__(self):
        super(MobileNetv1, self).__init__()
        self.skeleton_ = nn.Sequential(
            Conv(in_channels=3, out_channels=32, stride=2),
            DSConvBlock(in_channels=32, out_channels=64, stride=1),
            DSConvBlock(in_channels=64, out_channels=128, stride=2),
            DSConvBlock(in_channels=128, out_channels=128, stride=1),
            DSConvBlock(in_channels=128, out_channels=256, stride=2),
            DSConvBlock(in_channels=256, out_channels=256, stride=1),
            DSConvBlock(in_channels=256, out_channels=512, stride=2),
            # 5 x DSConvBlock (stride = 1)
            DSConvBlock(in_channels=512, out_channels=512, stride=1),
            DSConvBlock(in_channels=512, out_channels=512, stride=1),
            DSConvBlock(in_channels=512, out_channels=512, stride=1),
            DSConvBlock(in_channels=512, out_channels=512, stride=1),
            DSConvBlock(in_channels=512, out_channels=512, stride=1),
            # 7 * 7 * 512 => 7 * 7 * 1024
            DSConvBlock(in_channels=512, out_channels=1024, stride=2),
            # 7 * 7 * 1024 => 7 * 7 * 1024
            DSConvBlock(in_channels=1024, out_channels=1024, stride=1),
        )

        self.pool_ = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_ = nn.Linear(in_features=1024, out_features=1000)

        self.head_ = nn.Softmax()

    def forward(self, x):
        input_ = x
        temp_ = self.skeleton_(input_)

        temp_ = self.pool_(temp_)
        temp_ = temp_.view(temp_.size(0), -1)
        temp_ = self.fc_(temp_)
        result_ = self.head_(temp_)
        return result_
