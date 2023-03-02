import torch
import torch.nn
import numpy as np
from torchvision.ops import nms 

# YoloV3 Anchor size
ANCHORS = [
    [[116,60] , [156,198] , [373,326]], # 13*13大感受野使用的先验框
    [[30,61] , [62,45] , [59,119]], # 26*26中等感受野使用的先验框
    [[10,13] , [16,30] , [33,23]], # 52*52小感受野使用的先验框
]
    
def decode_box(self , org_shape , feature_inputs , num_classes=70):
    # 检测框属性等于类别 + x,y,w,h,confidence
    box_attr = num_classes + 5
    # 假设网络的输入为416*416*3，则feature_inputs的形状分别为
    # (b,75,13,13) (b,75,26,26) (b,75,52,52)
    for i , feature in enumerate(feature_inputs):
        b , c , h , w = feature.size()
        stride_h = org_shape[0] / h
        stride_w = org_shape[1] / w
        scaled_anchors = [(anchor_info[0] / stride_w, anchor_info[1] / stride_h) 
                          for anchor_info in ANCHORS[i]]
        # (b,75,13,13) => (b,3,75,13,13) => (b,3,13,13,75)
        predict_ = feature.view(b , len(ANCHORS[i]) , box_attr , h , w).permute(0,1,3,4,2).contiguous()
        # 获取先验框中心点调整参数
        x = torch.sigmoid(predict_[... , 0])
        y = torch.sigmoid(predict_[... , 1])
        # 获取先验框宽高调整参数
        w = predict_[... , 2]
        h = predict_[... , 3]
        # 获取置信度（是否有物体）
        conf = torch.sigmoid(predict_[... , 4])
        # 获取种类置信度
        pred_cls = torch.sigmoid(predict_[... , 5:])
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格
        grid_x = torch.linspace(0 , w - 1 , w). \
            repeat(h , 1).repeat(b * len(ANCHORS), 1 , 1).view(x.shape).type()
        grid_y = torch.linspace(0 , h - 1 , h). \
            repeat(w , 1).repeat(b * len(ANCHORS), 1 , 1).view(y.shape).type()
            
        # 按照网格生成先验框宽高
        anchor_w = torch.cuda.FloatTensor(scaled_anchors)
            
        
        
        
    return None