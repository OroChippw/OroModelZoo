import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_module import BaseModule

class Conv(BaseModule):
    def __init__(self):
        super(Conv, self).__init__()

    def forward(self, x):
        result_ = self.Conv2d(x)
        return result_


class ShuffleBlock(BaseModule):
    def __init__(self , in_channels , hidden_channels , out_channels , kernel_size , stride ,
                 group ,is_firstgroup : bool = False):
        super(ShuffleBlock, self).__init__()
        assert stride in [1,2], f'stride must in [1, 2]. ' \
                                 f'But received {stride}.'
        self.stride = stride
        self.group = group
        self.is_firstgroup = is_firstgroup

        layers = []


        branch_main_1 = []

        branch_main_2 = []

        if self.stride == 1 :
            layers.extend(
                # 1*1 GConv
                nn.Conv2d(in_channels=in_channels , out_channels=hidden_channels , kernel_size=1,
                          stride=stride , padding=0 , groups=1 if is_firstgroup else group,
                          bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            )
            pass
        elif self.stride == 2:
            if self.is_firstgroup:
                pass
            else :
                pass

        self.branch_ = nn.Sequential(*layers)

        if self.stride == 2:
            self.branch_pool = nn.AvgPool2d(kernel_size=3 , stride=2 , padding=1)

    def channel_shuffle(self , x):
        b , c , h , w = x.size()
        assert
        return x

    def forward(self, x):
        input_ = x
        result_ = ""
        if self.stride == 1 :
            return F.relu(input_ + result_)
        elif self.stride == 2 :
            return F.relu(torch.cat((self.branch_pool(input_) , result_) , dim=1))