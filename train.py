import torch
import os.path as osp

from models import MobileNetv1
from models import MobileNetv2



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() 
                            else 'cpu')
    a = torch.randn(2 , 3 , 224 , 224) # NCHW
    weight_path = r"weights/mobilenet_v2-b0353104.pth"
    assert osp.exists(weight_path) , \
            "file {} does not exist".format(weight_path)
    

    net = MobileNetv2()
    pretrained_weights = torch.load(weight_path , map_location=device)

    net.to(device)
    result_ = net(a)
    print(result_)
    print(result_.shape)


if __name__ == '__main__':
    main()
    