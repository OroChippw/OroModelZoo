import torch


def weights_init(model , init_type='normal' , init_gain=0.02):
    def init_func(m):
        classname = m._class_.__name__
        if hasattr(m , 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data , 0.0 , init_gain)
            elif init_type == 'xavier' :
                torch.nn.init.xavier_normal_(m.weight.data , gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data , a=0 , mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data , gain=init_gain)
            else : 
                raise NotImplementedError(f'Initialization method {init_type} is not implemented')
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data , 1.0 , 0.02)
            torch.nn.init.normal_(m.bias.data , 0.0)
    model.apply(init_func)
    print(f'Finsh Init model with {init_type}')
    