import torch
import torch.nn as nn

from .base_module import BaseModule
from ..builder import BACKBONES

class CBL_Residual_Unit(BaseModule):
    '''
        残差结构Residual Unit
        由两个CBL（Conv + Batch Normalization + LeakyReLU）组成
        一个CBL中为1*1卷积下降通道数、另一个为3*3卷积提取特征并通道升维
        最后接上一个残差边
    '''
    def __init__(self , in_channels , out_channels):
        self.CBL_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, 
                      kernel_size=1 , bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1)
        )
        self.CBL_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=3 , padding=1 , bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        input_ = x
        temp_ = self.CBL_1x1(input_)
        temp_ = self.CBL_3x3(temp_)
        result_ = input_ + temp_
        return result_

@BACKBONES.register_module()
class DarkNet53(BaseModule):
    def __init__(self , residual_loop , init_channels = 32) -> None:
        # residual_loop中以list的形式存储残差块循环多少次
        self.residual_loop = residual_loop
        # init_channels是第一个卷积的扩充通道数,init_Conv用来扩充通道维度
        self.init_channels = init_channels
        self.Block_out_channels = [64,128,256,512,1024]
        self.init_Conv = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=self.init_channels , kernel_size=3 , 
                      stride=1 , padding=1 , bias=False),
            nn.BatchNorm2d(self.init_channels) , 
            # torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
            # negative_slope：控制负斜率的角度，默认等于0.01 
            nn.LeakyReLU(negative_slope=0.1)
        )
        # Residual Block 1x 64
        # 416 * 416 * 32 => 208 * 208 * 64
        self.Residual_Block_1x_1 = self._make_block([32,64],replace_times=self.residual_loop[0])
        # Residual Block 2x 128
        # 208 * 208 * 64 => 104 * 104 * 128
        self.Residual_Block_2x_2 = self._make_block([64,128],replace_times=self.residual_loop[1])
        # Residual Block 8x 256
        # 104 * 104 * 128 => 52 * 52 * 256
        self.Residual_Block_8x_3 = self._make_block([128,256],replace_times=self.residual_loop[2])
        # Residual Block 8x 512
        # 52 * 52 * 256 => 26 * 26 * 512
        self.Residual_Block_8x_4 = self._make_block([256,512],replace_times=self.residual_loop[3])
        # Residual Block 4x 1024
        # 26 * 26 * 512 => 13 * 13 * 1024
        self.Residual_Block_4x_5 = self._make_block([512,1024],replace_times=self.residual_loop[4])
    
    def _make_block(self , channel_info , n_residuals):
        # channel_info是包含该Residual Block的输入通道数和输出通道数的列表
        layers_ = []
        # 在每一个Residual_Block里面，先用一个补偿为2的3*3卷积进行下采样
        layers_.append(nn.Conv2d(in_channels=channel_info[0] , out_channels=channel_info[1] , 
                                kernel_size=3 , stride=2 , padding=1))
        layers_.append(nn.BatchNorm2d(channel_info[1]))
        layers_.append(nn.LeakyReLU(0.1))
        
        for i in range(0 , n_residuals):
            layers_.append(CBL_Residual_Unit(channel_info[0] , channel_info[1]))
            
        return nn.Sequential(*layers_)
        
    
    def forward(self , x):
        # 假设输入为416*416*3
        input_ = x
        # 416 * 416 * 3 ==init_Conv=> 416 * 416 * 32
        temp_ = self.init_Conv(input_)
        # 416 * 416 * 32 ==Residual_Block_1x_1=> 208 * 208 * 64
        temp_ = self.Residual_Block_1x_1(temp_)
        # 208 * 208 * 64 ==Residual_Block_2x_2=> 104 * 104 * 128
        temp_ = self.Residual_Block_2x_2(temp_)
        # 104 * 104 * 128 ==Residual_Block_8x_3=> 52 * 52 * 256
        out_3 = self.Residual_Block_8x_3(temp_)
        # 52 * 52 * 256 ==Residual_Block_8x_4=> 26 * 26 * 512
        out_4 = self.Residual_Block_8x_4(out_3)
        # 26 * 26 * 512 ==Residual_Block_4x_5==> 13 * 13 * 1024
        out_5 = self.Residual_Block_4x_5(out_4)
        
        return out_3 , out_4 , out_5

if __name__ == '__main__':
    backbone = DarkNet53(residual_loop=[1,2,8,8,4])