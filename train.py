import argparse
import os
import torch
import torchvision
import logging
import time
import os.path as osp
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
from models import MobileNet_v3_large ,MobileNet_v3_small


logging.getLogger().setLevel(logging.INFO)

def args_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--epoches' ,
        type=int ,
        default=20 ,
        help='train epoches')
    parse.add_argument(
        '--batch_size' ,
        type=int ,
        default=16 ,
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

def train_one_epoch(model , optimizer , loss , lr_schedule , epoch , log_interval ,
                    dataloader , device , batch_size):
    start_time = time.time()
    all_loss = 0.0
    all_acc = 0
    model.train()
    for idx , (img , labels) in enumerate(dataloader):
        img = img.to(device)
        labels = labels.to(device)
        pred_ = model(img)
        loss_ = loss(pred_ , labels)

        optimizer.zero_grad()
        cur_acc = (pred_.data.max(dim = 1)[1]==labels).sum()

        all_acc += cur_acc

        if idx % log_interval == 0 :
            logging.info("epoch:{} iters:{}/{} loss:{} acc:{} lr:{}" \
                         .format(epoch,idx,len(dataloader),loss_.item(),cur_acc*100/len(labels),
                                 optimizer.param_groups[0]['lr']))

        lr_schedule.step(loss_.item())

    end_time = time.time()
    all_loss /= len(dataloader)
    # acc_ = all_acc *100 / len(dataloader) * batch_size
    return all_loss
def model_val(model , dataloader , epoch , device):
    start = time.time()
    model.eval()
    all_acc = 0
    for idx , (img , labels) in enumerate(dataloader):
        img = img.to(device)
        labels = labels.to(device)
        pred_ = model(img)

        cur_acc = (pred_.data.max(dim = 1)[1] == labels).sum() / len(labels)
        all_acc += cur_acc

    end_time = time.time()
    print("epoch:{} acc:{}".format(epoch,all_acc*100/len(dataloader)))
    return all_acc/len(dataloader)

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

    net = MobileNetv2(widen_factor=0.75 , num_classes=10).to(device)
    summary(net , (3,224,224))
    # net = MobileNet_v3_large()
    # net = MobileNet_v3_small()

    optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad],
                          lr=0.01, momentum=0.9, weight_decay=5e-4,
                          nesterov=True)
    loss = nn.CrossEntropyLoss()
    # lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                    factor=0.5, patience=200, min_lr=1e-6)
    lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=0.1,
                                                       last_epoch=-1)

    start_epoches = 0
    logging.info("Training...")

    for epoch in range(start_epoches , args.epoches):
        mean_loss = train_one_epoch(net , optimizer , loss , lr_schedule , epoch , args.log_interval ,
                                    train_dataloader ,device , args.batch_size)
        writer.add_scalar('train_loss' , mean_loss , epoch)

        val_acc = model_val(net , val_dataloader , epoch , device)
        writer.add_scalar('val_acc',val_acc,epoch)

        if (epoch + 1) % args.save_interval == 0 :
            weight_path_save_path = osp.join(args.save_dir , 'mobilenetv2_{}'.format(epoch))
            save_params = {
                'model':net.state_dict(),
                'epoch':epoch,
                'optim':optimizer.state_dict()
            }
            torch.save(save_params , weight_path_save_path)





    result_ = net(a)
    print(result_)
    print(result_.shape)


if __name__ == '__main__':
    main()
    