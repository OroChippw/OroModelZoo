import torch
import os.path as osp

from models import MobileNetv1
from models import MobileNetv2
from models import MobileNet_v3_large ,MobileNet_v3_small


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() 
                            else 'cpu')
    a = torch.randn(2 , 3 , 224 , 224) # NCHW
    # weight_path = r"weights/mobilenet_v2/mobilenet_v2-b0353104.pth"
    # assert osp.exists(weight_path) , \
    #         "file {} does not exist".format(weight_path)
    # pretrained_weights = torch.load(weight_path , map_location=device)

    net = MobileNetv2(widen_factor=0.75)
    # net = MobileNet_v3_large()
    # net = MobileNet_v3_small()

    net.to(device)
    result_ = net(a)
    print(result_)
    print(result_.shape)


if __name__ == '__main__':
    main()
    