import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# from ..base_module import BaseModule
# from ..builder import BACKBONE

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def forward(self, x):
        pass

    def train(self, mode=True):
        pass

class BaseConvBlock(BaseModule):
    def __init__(self , in_channels , out_channels , kernel_size , stride = 1):
        super().__init__()
        self.ConvBNPReLU = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=kernel_size , 
                        padding=int((kernel_size - 1) / 2 ) , bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels)
        )
    
    def forward(self, x):
        input_ = x
        temp_ = self.ConvBNPReLU(input_)
        result_ = temp_
        return result_

class local_feature_extractor(BaseModule):
    """
        employ channel-wise convolutions
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class surrounding_context_extractor(BaseModule):
    """
        employ channel-wise convolutions
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class joint_feature_extractor(BaseModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return super().forward(x)

class global_context_extractor(BaseModule):
    '''
    Func :
        a global average pooling layer to aggregate the global context corresponding to the purple region

    '''
    def __init__(self):
        super(global_context_extractor , self).__init__()
        self.branch_ = nn.Sequential(

        )

    def forward(self, x):
        input_ = x 
        temp_ = self.branch_s(input_)
        result_ = torch.cat((input_ , temp_) , dim=1)
        return result_

class CGBlock(BaseModule):
    """
    Func :
        Context Guided Block
          consist of a local feature extractor floc(∗), a surrounding context extractor fsur(∗),
        a joint feature extractor fjoi(∗) and a global context extractor fglo(∗)

    Args:
        BaseModule (_type_): _description_
    """
    def __init__(self , in_channels , out_channels):
        super(CGBlock , self).__init__()
        # floc(*)
        self.flo = local_feature_extractor()
        # fsur(*)
        self.fsur = surrounding_context_extractor()
        # fjoi(*)
        self.fjoi = joint_feature_extractor()
        # fglo(*)
        self.fglo = global_context_extractor()

    def forward(self, x):
        input_ = x
        temp_ = torch.cat(dim = 1)
        result_ = self.fglo(temp_)
        return result_

class InputInjection(BaseModule):
    def __init__(self , downsampling_ratio):
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.pool_ = nn.ModuleList()
        for i in range(0 , self.downsampling_ratio):
            self.pool_.append(
                nn.AvgPool2d(kernel_size=3 , stride=2 , padding=1)
            )

    def forward(self, x):
        input_ = x
        for pool in self.pool_:
            input_ = pool(input_)
        result_ = input_
        return result_

class CGStage(BaseModule):
    def __init__(self , num_blockes , in_channels , out_channels):
        super().__init__()
        self.num_blockes = num_blockes
        if self.num_blockes == 0 :
            self.BaseConvBlock_1 = BaseConvBlock(in_channels=in_channels , out_channels=out_channels ,
                                                kernel_size=3 , stride=2)
            self.BaseConvBlock_2 = BaseConvBlock(in_channels=out_channels , out_channels=out_channels ,
                                                kernel_size=3 , stride=1)
        else :
            self.Blockpipeline = nn.ModuleList()
            for i in range(self.num_blockes):
                self.Blockpipeline.append(
                    CGBlock(in_channels=in_channels , out_channels=out_channels)
                )

    def forward(self, x):
        input_ = x
        if self.num_blockes == 0 : # Stage 1 680 * 680 * 3 => 340 * 340 * 3
            for idx in range(3):
                if idx == 0 :
                    temp_ = self.BaseConvBlock_1(input_)
                else : 
                    temp_ = self.BaseConvBlock_2(temp_)
            result_ = temp_
        else : # Other stage
            for block in self.Blockpipeline :
                pass
        return result_

class CGNet(BaseModule):
    def __init__(self , stage2_M , stage3_N , in_channels ,dropout = False):
        super(CGNet , self).__init__()
        self.stage_1 = CGStage(num_blockes = 0)
        self.stage_2 = CGStage(num_blockes = stage2_M)
        self.stage_3 = CGStage(num_blockes = stage3_N)
        self.input_injection_1 = InputInjection(downsampling_ratio=1)
        self.input_injection_2 = InputInjection(downsampling_ratio=2)

    
    def forward(self, x):
        input_ = x
        input_downsample_2x = self.input_injection_1(input_)
        input_downsample_4x = self.input_injection_2(input_)
        # 680 * 680 * 3 => 340 * 340 * 32
        stage1_output = self.stage_1(input_)
        # 340 * 340 * 32 => 170 * 170 * 64
        stage2_output = self.stage_2(torch.cat((stage1_output , input_downsample_2x) , dim=1))
        # 170 * 170 * 64 => 85 * 85 * 128
        stage3_output = self.stage_3(torch.cat((stage2_output , input_downsample_4x) , dim=1))
        # 85 * 85 * 128 => 85 * 85 * 19
        temp_ = ""

        result_ = temp_
        return result_
    
class CGNet_M3N15(BaseModule):
    pass

class CGNet_M3N21(BaseModule):
    pass
