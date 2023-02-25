import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from .base_module import BaseModule
from .mobilenet_v2 import MobileNetv2
from .utils import ASPP
from ..builder import MODELS

@MODELS.register_module()
class DeepLabV3p(BaseModule):
    def __init__(self , num_classes=2 , backbone='Mobilenet' , 
                pretrained=False , down_sample_factor=16 , **cfg):
        super().__init__()
        assert backbone in ["Mobilenet" , "Xception"] , \
            f"Backbone only support `Mobilenet` or `Xception` , but got {backbone}"
        self.num_classes = num_classes
        self.pretrained = pretrained
        if backbone == "Xception":
            # 使用Xception作为主干特征提取器
            self.backbone_ = ""
            in_channels = 2048
            low_channels = 256
        elif backbone == "Mobilenet":
            # 使用MobileNetv2作为主干特征提取器
            self.backbone_ = MobileNetv2(pretrained=self.pretrained)
            in_channels = 320
            low_channels = 24
        
        # ASPP特征提取模块，利用不同膨胀率的膨胀卷积进行特征提取
        self.aspp_ = ASPP(in_channels=in_channels , out_channels=256 , rate=16//down_sample_factor)
        
        # Low-level Features浅层特征边
        self.lowlevel_conv = nn.Sequential(
            nn.Conv2d(low_channels , 48 , 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # 将经过ASPP特征提取后的主干部分上采样四倍后和浅层特征进行堆叠后利用卷积再特征提取
        self.concat_conv = nn.Sequential(
            nn.Conv2d(48 + 256 , 256 , 3 , stride=1 , padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Conv2d(256 , 256 , 3 , stride=1 , padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        self.cls_conv = nn.Conv2d(256 , num_classes , 1 , stride=1)
        
    
    def forward(self, x):
        input_ = x
        b , c , w , h = input_.size()
        low_level_features , main_feature = self.backbone_(input_)
        
        main_feature_ = self.aspp_(main_feature)
        # => Decoder
        low_level_features_ = self.lowlevel_conv(low_level_features)
        
        main_feature_ = F.interpolate(main_feature_ , 
                                    size=(low_level_features_.size(2) , low_level_features_.size(3)) , 
                                    mode='bilinear' , align_corners=True)
        concat_ = self.concat_conv(torch.cat((main_feature_ , low_level_features_) , dim=1))
        cls_ = self.cls_conv(concat_)
        result_ = F.interpolate(cls_ , size=(h,w) , mode='bilinear' , align_corners=True)
        
        return result_