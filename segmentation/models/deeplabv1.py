import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .base_module import BaseModule
from ..builder import MODELS


class DeepLabV1(BaseModule):
    def __init__(self , num_classes):
        super().__init__()
        self.skeleton_ = nn.Sequential(
            # VGG16 Block1
            # 224 * 224 * 3 => 112 * 112 * 64
            nn.Conv2d(in_channels=3 , out_channels=64 , kernel_size=3 , 
                        stride=1 , padding=1) ,
            nn.ReLU(True),
            nn.Conv2d(in_channels=64 , out_channels=64 , kernel_size=3 , 
                        stride=1 , padding=1) ,
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),
            # VGG16 Block2
            # 112 * 112 * 64 => 56 * 56 * 128
            nn.Conv2d(in_channels=64 , out_channels=128 , kernel_size=3 , 
                        stride=1 , padding=1) ,
            nn.ReLU(True),
            nn.Conv2d(in_channels=128 , out_channels=128 , kernel_size=3 , 
                        stride=1 , padding=1) ,
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),
            # VGG16 Block3
            # 56 * 56 * 128 => 28 * 28 * 256
            nn.Conv2d(in_channels=128 , out_channels=256 , kernel_size=3 , 
                        stride=1 , padding=1) ,
            nn.ReLU(True),
            nn.Conv2d(in_channels=256 , out_channels=256 , kernel_size=3 , 
                        stride=1 , padding=1) ,
            nn.ReLU(True),
            nn.Conv2d(in_channels=256 , out_channels=256 , kernel_size=3 , 
                        stride=1 , padding=1) ,
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),
            # VGG16 Block4
            # 28 * 28 * 256 => 28 * 28 * 512
            # Pooling stride 2 => 1 
            nn.Conv2d(in_channels=256 , out_channels=512 , kernel_size=3 , 
                        stride=1 , padding=1) ,
            nn.ReLU(True),
            nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=3 , 
                        stride=1 , padding=1) ,
            nn.ReLU(True),
            nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=3 , 
                        stride=1 , padding=1) ,
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3 , stride=1 , padding=1),
            # VGG16 Block5
            # 28 * 28 * 512 => 28 * 28 * 512 
            # Convoltuion dilation 1 => 2 and Pooling stride 2 => 1 
            nn.Conv2d(in_channels=256 , out_channels=512 , kernel_size=3 , 
                        stride=1 , padding=1 , dilation=2) ,
            nn.ReLU(True),
            nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=3 , 
                        stride=1 , padding=1 , dilation=2) ,
            nn.ReLU(True),
            nn.Conv2d(in_channels=512 , out_channels=512 , kernel_size=3 , 
                        stride=1 , padding=1 , dilation=2) ,
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3 , stride=1 , padding=1),

            # AvgPool
            nn.AvgPool2d(kernel_size=3 , stride=1 , padding=1) ,

            # FC1
            # 28 * 28 * 512 => 28 * 28 * 1024
            # padding 1 => 12 and diation 2 => 12
            nn.Conv2d(in_channels=512 , out_channels=1024 , kernel_size=3 , 
                        stride=1 , padding=12 , dilation=12) , 
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),

            # FC2
            # 28 * 28 * 1024 => 28 * 28* 1024
            nn.Conv2d(in_channels=1024 , out_channels=1024 , kernel_size=1 , 
                        stride=1 , padding=0) , 
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),

            # Last Convolution
            nn.Conv2d(in_channels=1024 , out_channels=num_classes , kernel_size=1 , 
                        stride=1)
        )

        self.num_classes = num_classes

    def forward(self, x):
        input_ = x
        temp_ = self.skeleton_(input_)
        result_ = F.interpolate(temp_ , scale_factor=8 , 
                    mode='bilinear' , align_corners=False)
        return result_