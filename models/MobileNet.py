import torch
import torch.nn as nn
import torch.nn.functional as F 

# Depthwise Separable Convolution，DSC
class DSConvBlock(nn.Module):
    def __init__(self , in_channels , out_channels , 
                stride : int = 1, padding : int = 0):
        super(ConvBlock, self).__init__()
        # Depthwise
        self.DepthwiseBlock = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = in_channels, 
                        kernel_size = 3 , groups=in_channels , bias=False),
            nn.BatchNorm2d(in_channels),
        )
        # Pointwise
        self.PointwiseBlock = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                        kernel_size = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self , x):
        input_ = x
        temp_ = F.relu(self.DepthwiseBlock(input_))
        result_ = F.relu(self.PointwiseBlock(temp_))
        return result_




print()