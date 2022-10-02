from audioop import bias
import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .base_module import BaseModule


def _make_divisible(channels , divisor , min_channels = None):
    '''
    Functions:
        ensures that all layers have a channel number that is divisible by 8
    '''
    if min_channels is None : 
        min_channels = divisor
    new_channels = max(min_channels, 
                    int(channels + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor

    return new_channels


class h_sigmoid(BaseModule):
    def __init__(self , inplace = True):
        super(h_sigmoid , self).__init__()
        self.relu6 = F.ReLU6(inplace = inplace)
    
    def forward(self , x):
        return self.relu6(x + 3) / 6


class h_swish(BaseModule):
    def __init__(self , inplace = True):
        super(h_swish , self).__init__()
        self.alpha = h_sigmoid(inplace = inplace)
    
    def forward(self , x):
        return x * self.alpha(x)

class Conv(BaseModule):
    def __init__(self , in_channels , out_channels , kernel_size = 3 , 
                stride :int = 1 , padding :int = 1 , groups : int = 1 , use_hs = None):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.Conv2d = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
            kernel_size = kernel_size , stride = stride , padding = padding , 
            groups = groups , bias = False),
            nn.BatchNorm2d(out_channels),
            h_swish if use_hs else nn.ReLU(inplace = True),
        )
    
    def forward(self , x) : 
        result_ = self.Conv2d(x)
        return result_


class SEBlock(BaseModule):
    """
    Squeeze-and-Excitation
    Args : 
        reduction : 全连接层节点缩小为输入feature map的倍数
    """
    def __init__(self , input_channels , reduction = 4):
        super(SEBlock , self).__init__()
        self.pool_ = nn.AdaptiveAvgPool2d(1)
        self.fc_ = nn.Sequential(
            nn.Linear(input_channels , _make_divisible(input_channels // reduction , 8)),
            nn.ReLU(inplace = True) , 
            nn.Linear(_make_divisible(input_channels // reduction , 8) , input_channels) , 
            h_sigmoid()
        )

    def forward(self , x):
        input_ = x
        b , c , _ , _ = input_.size()
        temp_ = self.pool_(input_).view(b , c)
        print(temp_.shape)
        result_ = self.fc_(temp_).view(b,c,1,1)
        return input_ * result_

class InvertedresBlock_SENet(BaseModule):
    def __init__(self , in_channels , hidden_channels , out_channels ,
                kernel_size , stride , padding , use_SE , use_hs):
        super(InvertedresBlock_SENet , self).__init__()
        assert stride in [1,2] , f'stride must in [1, 2]. ' \
            f'But received {stride}.'
        self.use_shortcut = (stride == 1) and (in_channels == out_channels)
        padding = (kernel_size - 1 ) // 2
        layers = []

        if in_channels == hidden_channels :
            layers.append(
                # Depthwise Conv
                Conv(hidden_channels , hidden_channels , kernel_size=kernel_size , 
                    stride = stride , padding = padding , groups=hidden_channels),
                # SEBlock
                SEBlock(hidden_channels) if use_SE else nn.Identity() , 
                # Linear Bottleneck : Pointwise Conv ,  is equivalent to a Linear 
                nn.Conv2d(hidden_channels , out_channels , kernel_size=1,
                        stride=1 , padding=0 , bias = False) , 
                nn.BatchNorm2d(out_channels)
            )
        else : 
            layers.append(
                # Pointwise Conv Ascension
                Conv(in_channels , hidden_channels , kernel_size=1 , 
                    stride = 1 , padding = 0),
                # Depthwise Conv
                Conv(hidden_channels , hidden_channels , kernel_size=kernel_size , 
                    stride = stride , padding = padding , groups=hidden_channels),
                # SEBlock
                SEBlock(hidden_channels) if use_SE else nn.Identity() , 
                # Linear Bottleneck : Pointwise Conv ,  is equivalent to a Linear 
                nn.Conv2d(hidden_channels , out_channels , kernel_size=1,
                        stride=1 , padding=0 , bias = False) , 
                nn.BatchNorm2d(out_channels)
            )

        self.Conv = nn.Sequential(*layers)


    def forward(self , x):
        input_ = x 
        if self.use_shortcut : 
            result_ = input_ + self.Conv(input_) 
        else :
            result_ = self.Conv(input_)
        return result_


def MobileNet_v3_large(**kwargs):
    '''
    Functions:
        Constructs a MobileNetV3-Large model
    '''
    # setting of bneck
    # k(kernel size) , t(expand_ratio) , c(output_channels) , SE(whether use SEBlock) , HS(whether use h_swish) , s(stride)
    cfgs = [
        [3,1,16,0,0,1],
        [3,4,24,0,0,2],
        [3,3,24,0,0,1],
        [5,3,40,1,0,2],
        [5,3,40,1,0,1],
        [5,3,40,1,0,1],
        [3,6,80,0,1,2],
        [3,2.5,80,0,1,1],
        [3,2.3,80,0,1,1],
        [3,2.3,80,0,1,1],
        [3,6,112,1,1,1],
        [3,6,112,1,1,1],
        [5,6,160,1,1,2],
        [5,6,160,1,1,1],
        [5,6,160,1,1,1],     
    ]
    return MobileNetv3(cfgs , mode = "large" , **kwargs)

def MobileNet_v3_small(**kwargs):
    '''
    Functions:
        Constructs a MobileNetV3-small model
    '''
    # setting of bneck
    # k(kernel size) , t(expand_ratio) , c(output_channels) , SE(whether use SEBlock) , HS(whether use h_swish) , s(stride)
    cfgs = [
        [3,1,16,1,0,2],
        [3,4.5,24,0,0,2],
        [3,3.67,24,0,0,1],
        [5,4,40,1,1,2],
        [5,6,40,1,1,1],
        [5,6,40,1,1,1],
        [5,3,48,1,1,1],
        [5,3,48,1,1,1],
        [5,6,96,1,1,2],
        [5,6,96,1,1,1],
        [5,6,96,1,1,1],
    ]
    return MobileNetv3(cfgs , mode = "small" , **kwargs)

class MobileNetv3(BaseModule):
    def __init__(self, cfg : list = None , mode : str = None , 
                widen_factor : float = 1.0 , num_classes : int = 1000, 
                pretrained  = None , ):
        super(MobileNetv3,self).__init__()
        assert mode in ["large" , "small"] , f"Mobilenetv3 is not support {mode} mode."
        self.cfg = cfg
        self.mode = mode 
        self.pretrained = pretrained
        
        # building first layer
        # modify the channels of the head convlution kernel , reduce 3ms
        in_channel = _make_divisible(16 * widen_factor , 
                                        divisor = 4 if widen_factor == 0.1 else 8)
        layers = []
        layers.append(
            nn.Conv2d(input_channels = 3 , out_channels=in_channel , kernel_size=3 ,  
                    stride = 2 , padding = 1 , bias = False),
            nn.BatchNorm2d(in_channel) ,
            h_swish() ,
        )
        for k , t , c , use_SE , use_hs ,s in self.cfg :
            output_channel = _make_divisible(c * widen_factor , 
                                            divisor = 4 if widen_factor == 0.1 else 8)
            exp_size = _make_divisible(in_channel * t , 
                                            divisor = 4 if widen_factor == 0.1 else 8)
            layers.append(InvertedresBlock_SENet(in_channel , exp_size , output_channel , kernel_size=k , 
                                        stride = s , use_SE = use_SE , use_hs = use_hs))
            in_channel = output_channel

        layers.append(
            nn.Conv2d(input_channels = in_channel , out_channels=exp_size , kernel_size=1 ,  
                    stride = 1 , padding = 0 , bias = False),
            nn.BatchNorm2d(in_channel) ,
            h_swish() ,
        )

        self.skeleton_ = nn.Sequential(*layers)

        self.pool_ = nn.AdaptiveAvgPool2d((1,1))

        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * widen_factor, 8) \
                        if widen_factor > 1.0 else output_channel[mode]

        self.head_ = nn.Sequential(
            nn.Linear(exp_size , output_channel) , 
            h_swish() , 
            nn.Linear(output_channel , num_classes) 
        )

        self._initialize_weights()
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2. / n))
                if m.bias is not None :
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m , nn.Linear):
                m.weight.data.normal_(0 , 0.01)
                m.bias.data.zero_()

    def forward(self , x):
        input_ = x 
        temp_ = self.skeleton_(input_)
        temp_ = self.pool_(input_)
        temp_ = temp_.view(temp_.size(0) , -1)
        result_ = self.head_(temp_)
        return result_