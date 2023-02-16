import torch
import torch.nn as nn


def weights_init(model , init_type='normal' , init_gain=0.02):
    def init_func(m):
        # print(f"m : {m} , type : {type(m)}" )
        if isinstance(m , nn.Conv2d):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data , 0.0 , init_gain)
                torch.nn.init.normal_(m.bias.data , 0.0 , init_gain)
            elif init_type == 'xavier' :
                torch.nn.init.xavier_normal_(m.weight.data , gain=init_gain)
                torch.nn.init.xavier_normal_(m.bias.data , gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data , a=0 , mode='fan_in')
                torch.nn.init.kaiming_normal_(m.bias.data , a=0 , mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data , gain=init_gain)
                torch.nn.init.orthogonal_(m.bias.data , gain=init_gain)
            else : 
                raise NotImplementedError(f'Initialization method {init_type} is not implemented')
        elif isinstance(m , nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data , 1.0 , 0.02)
            torch.nn.init.normal_(m.bias.data , 0.0)
    # model.apply(fn) will recursively apply the function fn to each submodule of the parent module
    model.apply(init_func)
    print(f'==> Finsh Initial model in method {init_type}')

# recommend

# def weights_init(m): 
#     if isinstance(m, nn.Conv2d): 
#         nn.init.xavier_normal_(m.weight.data) 
#         nn.init.xavier_normal_(m.bias.data)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight,1)
#         nn.init.constant_(m.bias, 0)
 