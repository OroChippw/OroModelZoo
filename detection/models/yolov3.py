import torch
import torch.nn as nn

from .yolov3_darknet53 import DarkNet53
from .base_module import BaseModule
from ..builder import MODELS

@MODELS.register_module()
class YoloV3(BaseModule):
    def __init__(self , num_classes , backbone='DarkNet53'):
        super().__init__()
        assert backbone in ["DarkNet53" , "MobileNetV2"] , \
            f"Backbone Only support `DarkNet53`,`MobileNetV2` but got {backbone}"
        self.num_classes = num_classes
        if backbone == "DarkNet53":
            self.backbone = DarkNet53(residual_loop=[1,2,8,8,4])
        
        
        self.yolo_head_1 = ""
        
        
    def forward(self, x):
        input_ = x
        # 假设输入为416*416*3，从主干特征网络中一次得到高维特征和低维特征
        # feature_1 : 52*52*256 , feature_2 : 26*26*512 , feature_3 : 13*13*512
        feature_1 , feature_2 , feature_3 = self.backbone(input_) 
        
        return super().forward(x)
    
    

if __name__ == '__main__':
    model = YoloV3()