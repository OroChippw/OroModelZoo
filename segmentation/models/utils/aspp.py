import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self , in_channels , out_channels , rate=1 , bn_momentum=0.1):
        super().__init__()
        '''
            bn_momentum决定最新计算的mini-batch中的均值mean_new和标准差var_new和
        上一个mini-batch保存的均值mean_last和标准差取多少比例
            mean = (1 - bn_momentum) * mean_new + (bn_momentum) * mean_last
            var = (1 - bn_momentum) * var_new + (bn_momentum) * var_last
        '''
        # 1*1 Conv
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , 1 , 1 , padding=0 , 
                      dilation=6 * rate , bias=True),
            nn.BatchNorm2d(out_channels , momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        # 3*3 Conv , rate = 6
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , 3 , 1 , padding=6 * rate , 
                      dilation=rate , bias=True),
            nn.BatchNorm2d(out_channels , momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        # 3*3 Conv , rate = 12
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , 3 , 1 , padding=12 * rate , 
                      dilation=12 * rate , bias=True),
            nn.BatchNorm2d(out_channels , momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        # 3*3 Conv , rate = 18
        self.branch_4 = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , 3 , 1 , padding=18 * rate , 
                      dilation=18 * rate , bias=True),
            nn.BatchNorm2d(out_channels , momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )
        # Global Average Pooling + Conv
        self.branch_5_conv = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , 1 , 1 , 0 , bias=True),
            nn.BatchNorm2d(out_channels , momentum=bn_momentum), 
            nn.ReLU(inplace=True)
        )
        # Concat + 1*1 Conv
        self.concat_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5 , out_channels , 1 , 1 , 0 , bias=True),
            nn.BatchNorm2d(out_channels , momentum=bn_momentum), 
            nn.ReLU(inplace=True)
        )
        
    def forward(self , x):
        input_ = x
        b , c , h , w = input_.size()
        branch_1_conv1x1 = self.branch_1(input_)
        branch_2_conv3x3_r6 = self.branch_2(input_)
        branch_3_conv3x3_r12 = self.branch_3(input_)
        branch_4_conv3x3_r18 = self.branch_4(input_)
        # branch_5 include GAP and 1*1 Conv
        global_feature = torch.mean(input_ , 2 , True)
        global_feature = torch.mean(global_feature , 3 , True)
        global_feature = self.branch_5_conv(global_feature)
        global_feature = F.interpolate(global_feature , (w , h) , mode='bilinear' , align_corners=True)
        # Concat the contents of the five branches and use 1*1 Conv intergration feature
        concat_feature = torch.cat([
            branch_1_conv1x1 , branch_2_conv3x3_r6 , branch_3_conv3x3_r12 , branch_4_conv3x3_r18 , global_feature
        ] , dim=1)
        result_ = self.concat_conv(concat_feature)
        return result_
