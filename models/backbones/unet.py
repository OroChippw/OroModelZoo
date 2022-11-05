import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_module import BaseModule
from ..builder import BACKBONE

class EncoderBlock(BaseModule):
    def __init__(self , in_channels , out_channels , use_maxpool = False):
        super(EncoderBlock, self).__init__()
        self.use_maxpool = use_maxpool
        self.baseConvBlock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels ,
                      kernel_size=3 , stride=1 , padding=1) ,
            nn.BatchNorm2d(out_channels) ,
            nn.ReLU() ,
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.downsample_byMaxPool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.downsample_byConv2d = nn.Sequential(
            nn.Conv2d(in_channels=out_channels , out_channels=out_channels ,
                      kernel_size=3 , stride=2 , padding=1),
            nn.BatchNorm2d(out_channels) ,
            nn.ReLU()
        )

    def forward(self, x):
        input_ = x
        temp_ = self.baseConvBlock(input_)
        result_1 = temp_
        if self.use_maxpool :
            temp_ = self.downsample_byMaxPool(temp_)
        else :
            temp_ = self.downsample_byConv2d(temp_)
        result_2 = temp_
        return result_1 , result_2

class DecoderBlock(BaseModule):
    def __init__(self , in_channels , out_channels):
        super(DecoderBlock, self).__init__()
        self.baseConvBlock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
        )
        self.upsample_ = nn.Sequential(
            # output = (input - 1) * stride + outputpadding - 2 * padding + kernelsize
            nn.ConvTranspose2d(in_channels=out_channels * 2 , out_channels=out_channels ,
                               kernel_size=3 , stride=2 , padding=1 , output_padding=1) ,
            nn.BatchNorm2d(out_channels) ,
            nn.ReLU() ,
        )

    def forward(self, x1 , encode_):
        input_ = x1
        temp_ = self.baseConvBlock(input_)
        temp_ = self.upsample_(temp_)
        result_ = torch.cat((temp_ , encode_) , dim=1)
        return result_


@BACKBONE.register_module(BaseModule)
class Unet(BaseModule):
    def __init__(self):
        super(Unet, self).__init__()
        output_channels = [64,128,256,512,1024]

        self.encoder_ = nn.Sequential(

        )

        self.decoder_ = ""

    def forward(self, x):
        pass
