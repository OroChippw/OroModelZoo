import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import MODELS

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def forward(self, x):
        pass

    def train(self, mode=True):
        pass

class EncoderBlock(BaseModule):
    def __init__(self , in_channels , out_channels , use_maxpool = True):
        super(EncoderBlock, self).__init__()
        self.use_maxpool = use_maxpool
        self.baseConvBlock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels ,
                      kernel_size=3 , stride=1) ,
            nn.BatchNorm2d(out_channels) ,
            nn.ReLU() ,
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.downsample_byMaxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.downsample_byConv2d = nn.Sequential(
            nn.Conv2d(in_channels=out_channels , out_channels=out_channels ,
                      kernel_size=3 , stride=2 ,  padding=1),
            nn.BatchNorm2d(out_channels) ,
            nn.ReLU()
        )

    def forward(self, x):
        input_ = x
        temp_ = self.baseConvBlock(input_)
        result_1 = temp_
        if self.use_maxpool :
            temp_ = self.downsample_byMaxPool(temp_)
        else :
            temp_ = self.downsample_byConv2d(temp_)
        result_2 = temp_
        return result_1 , result_2

class DecoderBlock(BaseModule):
    def __init__(self , in_channels , out_channels):
        super(DecoderBlock, self).__init__()
        self.baseConvBlock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * 2,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels * 2,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
        )
        self.upsample_ = nn.Sequential(
            # output = (input - 1) * stride + outputpadding - 2 * padding + kernelsize
            nn.ConvTranspose2d(in_channels=out_channels * 2 , out_channels=out_channels ,
                               kernel_size=3 , stride=2 , padding=1 , output_padding=1) ,
            nn.BatchNorm2d(out_channels) ,
            nn.ReLU() ,
        )
    

    def concat_(self , input_1 , input_2):
        '''
            copy and crop
        '''
        shape_1 = input_1.size()[3]
        shape_2 = input_2.size()[3]
        tensor_1 , tensor_2 = (input_1 , input_2) if shape_1 >= shape_2 else (input_2 , input_1)
        crop_ = int((tensor_1.size()[3] - tensor_2.size()[3]) / 2)
        tensor_1 = tensor_1[: , : , crop_:tensor_1.size()[3]-crop_ : , 
                            crop_:tensor_1.size()[3]-crop_]
        result_ = torch.cat((tensor_1 , tensor_2) , dim=1)
        return result_


    def forward(self, x1 , encode_):
        input_ = x1
        temp_ = self.baseConvBlock(input_)
        temp_ = self.upsample_(temp_)
        result_ = self.concat_(temp_ , encode_)
        return result_

class OutputHead(BaseModule):
    def __init__(self , in_channels , out_channels , final_channels):
        super(OutputHead , self).__init__()
        self.baseConvBlock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels, 
                      kernel_size=3 , stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels , out_channels=out_channels, 
                      kernel_size=3 , stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels , out_channels=final_channels , 
                    kernel_size=1 , stride=1),
            nn.BatchNorm2d(final_channels),
        )
        
    def forward(self, x):
        input_ = x 
        temp_ = self.baseConvBlock(input_)
        result_ = torch.sigmoid(temp_)
        return result_

@MODELS.register_module()
class UNet(BaseModule):
    def __init__(self , num_classes=2 , pretrained=None , **cfg):
        super(UNet, self).__init__()
        self.pretrained = pretrained
        output_channels = [64,128,256,512,1024]

        self.encoder_0 = EncoderBlock(3 , output_channels[0])
        self.encoder_1 = EncoderBlock(output_channels[0] , output_channels[1])
        self.encoder_2 = EncoderBlock(output_channels[1] , output_channels[2])
        self.encoder_3 = EncoderBlock(output_channels[2] , output_channels[3])

        self.decoder_3 = DecoderBlock(output_channels[3] , output_channels[3])
        self.decoder_2 = DecoderBlock(output_channels[4] , output_channels[2])
        self.decoder_1 = DecoderBlock(output_channels[3] , output_channels[1])
        self.decoder_0 = DecoderBlock(output_channels[2] , output_channels[0])

        self.head_ = OutputHead(output_channels[1] , output_channels[0] , final_channels=num_classes)


    def forward(self, x):
        input_ = x
        # Encoder
        # 572 * 572 * 3 => 568 * 568 * 64 / 284 * 284 * 64
        concat_0 , downsample_0 = self.encoder_0(input_)
        # 284 * 284 * 64 => 280 * 280 * 128 / 140 * 140 * 128
        concat_1 , downsample_1 = self.encoder_1(downsample_0)
        # 140 * 140 * 128 => 136 * 136 * 256 / 68 * 68 * 256
        concat_2 , downsample_2 = self.encoder_2(downsample_1)
        # 68 * 68 * 256 => 64 * 64 * 512 / 32 * 32 * 512
        concat_3 , downsample_3 = self.encoder_3(downsample_2)

        # Decoder
        # 32 * 32 * 512 / 64 * 64 * 512 => 56 * 56 * 1024
        upsample_3 = self.decoder_3(downsample_3  , concat_3)
        # 56 * 56 * 1024 / 136 * 136 * 256 => 104 * 104 * 512
        upsample_2 = self.decoder_2(upsample_3 , concat_2)
        # 104 * 104 * 512 / 280 * 280 * 128 => 200 * 200 * 256
        upsample_1 = self.decoder_1(upsample_2 , concat_1)
        # 200 * 200 * 256 / 568 * 568 * 64 => 392 * 392 * 128
        upsample_0 = self.decoder_0(upsample_1 , concat_0)

        # Head
        result_ = self.head_(upsample_0)

        return result_


if __name__ == '__main__':
    model = UNet(num_classes=2)
    # print(model)
    # summary(model , (3,572,572))