import time
import yaml
import argparse
import os , os.path as osp
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader , Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils import (mkdir_or_exist , init_dist_pytorch , get_dist_info , 
    build_multi_process  , weights_init)



def args_parse():
    parser = argparse.ArgumentParser(description="")
    # Model trainning settings
    parser.add_argument("--config",type=str,help="train config file path")
    # parser.add_argument("--checkpoint",type=str,help="initial weights path")
    # parser.add_argument("--start_epoches",type=int,default=0,help="start training epoches")
    # parser.add_argument("--num_epoches",type=int,default=2000,help="total training epoches")
    # parser.add_argument("--batch-size","-b",type=int,default=64,help="")
    parser.add_argument("--fp16",type=bool,default=False,action="store_true", help="")
    # parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # parser.add_argument("--work-dir",type=str,default="",help="")
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

    if args.config is not None:
        assert osp.isfile(args.config) and args.config.endswith('.yaml') , \
            f"args.config should be a yaml file , Instead of {osp.splitext(args.config)[-1]}"
        with open(args.config , 'r' , errors='ignore') as f:
            temp_ = yaml.load(f.read())
            print(temp_)
        for k , v in temp_.item():
            setattr(args , k , v)
        
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
    
    # build datasets


    # SyncBN is not support in DP mode
    if distributed_:
        train_sampler = DistributedSampler(dataset="" , shuffle=True)
        val_sampler = DistributedSampler(dataset="" , shuffle=False)

    train_dataloader = DataLoader("" , batch_size=args.batch_size)

    # Init Automatic mixed precision GradScaler
    if args.fp16:
        # torch.cuda.amp provides users with a more convenient mixed-precision training mechanism.
        # Users do not need to manually convert the model parameter dtype, amp will automatically select the appropriate numerical precision for the operator
        # For the problem of FP16 gradient value overflow during backpropagation, amp provides a gradient scaling operation, 
        # and before the optimizer updates the parameters, it will automatically unscale the gradient, so there will be no hyperparameters for model optimization. 
        from torch.cuda.amp import GradScaler as GradScaler
        scaler_ = GradScaler()
    else :
        scaler_ = None


    # Init tensorboard summarywriter
    writer_ = SummaryWriter('./logs')

    model_ = None

    if args.checkpoint is None:
        weights_init(model_)
    else : 
        # Load according to the Key of the pre-trained weight and the Key of the model
        model_dict_ = model_.state_dict()
        pretrain_dict_ = torch.load(args.checkpoint , map_location=args.device)
        success_load , fail_load , temp_dict = [] , [] , {}
        for k , v in pretrain_dict_.items():
            if k in model_dict_.keys() and np.shape(model_dict_[k]) == np.shape(v):
                temp_dict[k] = v
                success_load.append(k)
            else : 
                fail_load.append(k)
        model_dict_.update(temp_dict)
        model_.load_state_dict(model_dict_)
        if rank == 0:
            print("Successfully Load Key : " , str(success_load))
            print("Fail Load Key : " , str(fail_load))
        
    if distributed_:
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_)

    # loss function
    criterion_ = nn.CrossEntropyLoss()
    # optimizer 
    if args.TRAIN.optimizer == 'Adam':
        optimizer_ = optim.Adam(model_.parameters() , lr=args.TRAIN.LR)
    elif args.TRAIN.optimizer == 'SGD':
        optimizer_ = optim.SGD(model_.parameters() , lr=args.TRAIN.LR , momentum=0.9 , weight_decay=0.0001)
    if args.TRAIN.multistep_lr:
        lr_scheduler_ = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_ , milestones=args.TRAIN.LR_STEP , 
                            gamma=args.TRAIN.LR_FACTOR)

    metric_ = None

    for epoch in range(args.start_epoches , args.num_epoches):
        model_.train()
        optimizer_.zero_grad()
        loss_ = criterion_()
        if scaler_ is not None : 
            scaler_.scale(loss=loss_).backward()
            scaler_.step(optimizer=optimizer_)
            scaler_.update()
        else : 
            loss_.backward()
            optimizer_.step()
        writer_.add_scalar("loss function" , loss_.item() , epoch)
        

        # Save model last ckpt , best ckpt and delete the value ckpt
        torch.save()

    writer_.close()



if __name__ == '__main__':
    main()
    