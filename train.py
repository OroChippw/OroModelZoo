import argparse
import os
import sys

import torch
import torchvision
import logging
import time
import os.path as osp
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from torchsummary import summary
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import MobileNetv1
from models import MobileNetv2
# from models import MobileNet_v3_large ,MobileNet_v3_small


logging.getLogger().setLevel(logging.INFO)

def args_parse():
    parse = argparse.ArgumentParser(description="Train model args")
    parse.add_argument(
        '--epoches' ,
        type=int ,
        default=20 ,
        help='train epoches')
    parse.add_argument(
        '--batch_size' ,
        type=int ,
        default=24 ,
        help='batch size')
    parse.add_argument(
        '--weights' ,
        type=str ,
        help='model weight path')
    parse.add_argument(
        '--save_dir' ,
        type=str ,
        help='model save path')
    parse.add_argument(
        '--num_worker',
        type=int,
        default=4,
        help='')
    parse.add_argument(
        '--log_interval',
        type=int,
        default=5,
        help='')
    parse.add_argument(
        '--save_interval',
        type=int,
        default=5,
        help='')

    args = parse.parse_args()
    return args

def main():
    args = args_parse()

    if not osp.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.weights is not None :
        weight_path = args.weights
        assert osp.exists(weight_path) , \
                "file {} does not exist".format(weight_path)
        if weight_path.endswith('.pt') or weight_path.endswith('.pth'):
            ckpt = torch.load(weight_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() 
                            else 'cpu')
    logging.info("using {} to train".format(device))

    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # 将numpy数组或PIL.Image读的图片转换成(C,H, W)的Tensor格式且/255归一化到[0,1.0]之间
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    a = torch.randn(2 , 3 , 224 , 224) # NCHW

    # Download Datasets
    train_datasets = torchvision.datasets.CIFAR10('data' ,
                      train=True ,transform=transform,download=True)
    val_datasets = torchvision.datasets.CIFAR10('data' ,
                      train=False , transform=transform,download=True)

    # Load Datasets
    train_dataloader = DataLoader(train_datasets , batch_size=args.batch_size ,
                                  shuffle=True , num_workers=args.num_worker , pin_memory=False)
    val_dataloader = DataLoader(val_datasets , batch_size=args.batch_size ,
                                  shuffle=False , num_workers=args.num_worker , pin_memory=False)
    logging.info("Loading Datasets...")
    writer = SummaryWriter('events')

    net = MobileNetv2(widen_factor=1.0 , num_classes=10).to(device)
    # summary(net , (3,224,224))
    # net = MobileNet_v3_large()
    # net = MobileNet_v3_small()

    optimizer = optim.SGD(net.parameters() , lr = 0.001 ,momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    # lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                    factor=0.5, patience=200, min_lr=1e-6)
    # lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.1,
    #                                                    last_epoch=-1)

    logging.info("Training...")

    best_acc = 0.0
    for epoch in range(args.epoches):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader , file=sys.stdout)
        for idx , data in enumerate(train_bar):
            images , labels = data
            optimizer.zero_grad()# 清空之前的梯度信息进行正向传播
            pred_ = net(images.to(device))
            loss_ = loss_func(pred_ , labels.to(device))
            loss_.backward()# 将得到的损失反向传播到每个节点中
            optimizer.step()# 更新每个节点的参数

            running_loss += loss_.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}" \
                .format(epoch + 1,args.epoches,loss_)

        writer.add_scalar('train_loss' , running_loss , epoch)

        net.eval()
        eval_acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader , file=sys.stdout)
            for val_data in val_bar:
                images , labels = val_data
                output_ = net(images.to(device))
                predict_ = torch.max(output_ , dim=1)[1]
                eval_acc += torch.eq(predict_ , labels.to(device)).sum().item()
        val_acc = eval_acc / len(val_datasets)
        writer.add_scalar('val_acc',val_acc,epoch)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / len(train_dataloader), val_acc))

        if (epoch + 1) % args.save_interval == 0 and val_acc > best_acc:
            best_acc = val_acc
            weight_path_save_path = osp.join(args.save_dir , 'mobilenetv2_{}.pth'.format(epoch + 1))
            torch.save(net.state_dict() , weight_path_save_path)

    logging.info("Finish Training")



    # result_ = net(a)
    # print(result_)
    # print(result_.shape)


if __name__ == '__main__':
    main()
    