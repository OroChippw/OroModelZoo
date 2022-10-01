import torch
from models import MobileNetv1



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() 
                            else 'cpu')
    a = torch.randn(2 , 3 , 224 , 224) # NCHW
    print(a.shape)
    net = MobileNetv1()
    result_ = net(a)
    print(result_)
    print(result_.shape)


if __name__ == '__main__':
    main()
    