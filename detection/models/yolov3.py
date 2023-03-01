import torch
import torch.nn as nn

from .yolov3_darknet53 import DarkNet53
from .base_module import BaseModule
from ..builder import MODELS

class CBL_Block(BaseModule):
    def __init__(self , in_channels , out_channels , kernel_size):
        super().__init__()
        pad_ = (kernel_size - 1) // 2 if kernel_size else 0
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , kernel_size , 
                      stride=1 , padding=pad_ , bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.LeakyReLU(0.01)
        )
        
    def forward(self, x):
        input_ = x
        result_ = self.ConvBlock(input_)
        return result_

@MODELS.register_module()
class YoloV3(BaseModule):
    def __init__(self , num_classes , backbone='DarkNet53' , anchors_num=3):
        super().__init__()
        assert backbone in ["DarkNet53" , "MobileNetV2"] , \
            f"Backbone Only support `DarkNet53`,`MobileNetV2` but got {backbone}"
            
        self.num_classes = num_classes
        # 假设为VOC数据集则20个类,  75 = 3*25 = 3*（20+1+4）
        # 假设为COCO数据集则80个类,  225 = 3*85 = 3*（80+1+4）
        self.head_channels = anchors_num * (self.num_classes + 5) # 5 : x,y,w,h,confidence[0/1]
        
        if backbone == "DarkNet53":
            self.backbone = DarkNet53(residual_loop=[1,2,8,8,4])
            
        self.backboneoutput_fliters = self.backbone.Block_out_channels # [64, 128, 256, 512, 1024]
        
        # 最低分辨率特征图进入的连续五个CBL和对应的Yolo Head，FPN的顶层
        self.layer_1 = self._make_5xCBL_yolohead(in_channels=self.backboneoutput_fliters[-1] , 
                                                 fliters_list=[512,1024] , head_outchannels=self.head_channels)
        # 中等分辨率特征图进入的FPN的中层，得先经过一个卷积改变通道维度，再进行上采样
        self.layer_2_Conv = CBL_Block(in_channels=512 , out_channels=256 , kernel_size=1)
        self.upsample_ = nn.Upsample(scale_factor=2 , mode='nearest')
        self.layer_2 = self._make_5xCBL_yolohead(in_channels=self.backboneoutput_fliters[-1] + 256 , 
                                                 fliters_list=[256,512] , head_outchannels=self.head_channels)
        # 最高分辨率的特征图进入的FPN的底层，同样得先经过一个卷积改变通道维度，再进行上采样（和上面复用不作冗余实现）
        self.layer_3_Conv = CBL_Block(in_channels=256 , out_channels=128 , kernel_size=1)
        self.layer_3 = self._make_5xCBL_yolohead(in_channels=self.backboneoutput_fliters[-2] + 128 , 
                                                 fliters_list=[128,256] , head_outchannels=self.head_channels)
        
        
    def _make_5xCBL_yolohead(self , in_channels , fliters_list , head_outchannels):
        layer_ = nn.Sequential(
            # 假设输入的为13*13*1024，则in_channels为1024，对应位置的CBL通道为1024，经过Block后希望得到的是13*13*512
            # 前五个卷积为CBL*5，用于特征提取
            CBL_Block(in_channels , fliters_list[0] , 1),
            CBL_Block(fliters_list[0] , fliters_list[1] , 3),
            CBL_Block(fliters_list[1] , fliters_list[0] , 1),
            CBL_Block(fliters_list[0] , fliters_list[1] , 3),
            CBL_Block(fliters_list[1] , fliters_list[0] , 1),
            # Yolo Head本质上是一次3x3卷积加上一次1x1卷积，3x3卷积的作用是特征整合，1x1卷积的作用是调整通道数
            CBL_Block(fliters_list[0] , fliters_list[1] , 3),
            nn.Conv2d(fliters_list[1] , head_outchannels , 1 , stride=1 , padding=0 , bias=True)
        )
        return layer_
        
    
    def forward(self, x):
        input_ = x
        # 假设输入为416*416*3，从主干特征网络中一次得到高维特征和低维特征
        # feature_x序号依次从高分辨率到低分辨率
        # feature_3 : 52 * 52 * 256 , feature_2 : 26 * 26 * 512 , feature_1 : 13 * 13 * 512
        feature_1 , feature_2 , feature_3 = self.backbone(input_) 
        
        # ----------------------------------------------------- #
        # 13 * 13 * 1024的特征层[FPN-1]
        # 13 * 13 * 1024 => 13 * 13 * 512 => 13 * 13 * 1024 => 13 * 13 * 512 => 13 * 13 * 1024 => 13 * 13 * 512
        layer_1_outbranch = self.layer_1[:5](feature_1)
        yolo_head_1 = self.layer_1[5:](layer_1_outbranch) # 13 * 13 * 75
        # 13 * 13 * 512 => 13 * 13 * 256 => 26 * 26 * 256
        layer_2_in = self.layer_2_Conv(layer_1_outbranch)
        layer_2_in = self.upsample_(layer_2_in)
        # 26 * 26 * 256 + 26 * 26 * 512 = 26 * 26 * 768
        layer_2_in = torch.cat([layer_2_in , feature_2] , dim=1)
        
        # ----------------------------------------------------- #
        # 26 * 26 * 512的特征层[FPN-2]
        # 26 * 26 * 768 => 26 * 26 * 256 => 26 * 26 * 512 => 26 * 26 * 256 => 26 * 26 * 512 => 26 * 26 * 256
        layer_2_outbranch = self.layer_2[:5](layer_2_in)
        yolo_head_2 = self.layer_2[5:](layer_2_outbranch) # 26 * 26 * 75
        # 26 * 26 * 256 => 26 * 26 * 128 => 52 * 52 * 128
        layer_3_in = self.layer_3_Conv(layer_2_outbranch)
        layer_3_in = self.upsample_(layer_3_in)
        # 52 * 52 * 128 + 52 * 52 * 256 = 52 * 52 * 384
        layer_3_in = torch.cat([layer_3_in , feature_3] , dim=1)
        
        # ----------------------------------------------------- #
        # 52 * 52 * 256的特征层[FPN-3]
        yolo_head_3 = self.layer_3(layer_3_in) # 52 * 52 * 75
        
        # 如果是VOC数据集，则最终的输出应该为三个shape为(N,13,13,75)，
        # (N,26,26,75)，(N,52,52,75)的数据，对应每个图分为13x13、26x26、52x52的网格上3个先验框的位置。
        output_1 = yolo_head_1
        output_2 = yolo_head_2
        output_3 = yolo_head_3

        return output_1 , output_2 , output_3
    
    

if __name__ == '__main__':
    model = YoloV3()