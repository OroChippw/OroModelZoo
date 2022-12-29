import argparse
import os , os.path as osp
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader , Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils import (mkdir_or_exist , init_dist_pytorch , get_dist_info , build_multi_process)



def args_parse():
    parser = argparse.ArgumentParser(description="")
    # Model trainning settings
    parser.add_argument("--config",type=str,help="train config file path")
    parser.add_argument("--checkpoint",type=str,default="",help="initial weights path")
    parser.add_argument("--start_epoches",type=int,default=0,help="start training epoches")
    parser.add_argument("--num_epoches",type=int,default=2000,help="total training epoches")
    parser.add_argument("--batch-size","-b",type=int,default=64,help="")
    parser.add_argument("--fp16",type=bool,default=False,action="store_true", help="")
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument("--work-dir",type=str,default="",help="")
    # Distributed Data Parallel settings
    parser.add_argument("--device",default="",help="cuda device or cpu")
    parser.add_argument("--num_worker",type=int,default=4,help="")
    parser.add_argument("--gpu_workers",type=int,default=1,help="")
    parser.add_argument("--backend",type=str,default='nccl',help="")
    parser.add_argument("--launcher",choices=['none','pytorch'],default='pytorch',help='job launcher')
    parser.add_argument("--local-rank",type=int,default=-1,help="")


    args = parser.parse_args()
    return args

def main():
    args = args_parse()

    if args.work_dir is None :
        args.work_dir = osp.join('./work_dirs' , osp.splitext(osp.basename(args.config))[0])
    mkdir_or_exist(args.work_dir)
    
    # set cudnn_benchmark to accelerate the network. 
    # The applicable scenario is that the network structure is fixed (not dynamic)
    torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        distributed_ = False
    else :
        distributed_ = True
        init_dist_pytorch(args.backend)
        rank , world_size = get_dist_info()
    
    # init logger
    timestamp = time.strftime('%Y%m%d_%H%M%S' ,time.localtime())
    log_file = osp.join(args.work_dir , f'{timestamp}.log')

    # build multi-process settings
    build_multi_process(args.gpu_)
    
    # SyncBN is not support in DP mode
    if distributed_:
        train_sampler = DistributedSampler(dataset="" , shuffle=True)

    train_dataloader = DataLoader("" , batch_size=args.batch_size)

    if args.fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else :
        scaler = None


    # Init tensorboard summarywriter
    writer_ = SummaryWriter('./logs')

    model_ = None
    # loss function
    crit_ = nn.CrossEntropyLoss()
    # optimizer 
    optimizer_ = optim.Adam()
    metric_ = None
    for epoch in range(args.start_epoches , args.num_epoches):
        model_.train()
        optimizer_.zero_grad()
        loss_ = crit_()
        loss_.backward()
        optimizer_.step()
        writer_.add_scalar("loss function" , loss_.item() , epoch)
        

        # Save model last ckpt , best ckpt and delete the value ckpt
        torch.save()

    writer_.close()



if __name__ == '__main__':
    main()
    