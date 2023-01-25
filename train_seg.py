import time
import yaml
import argparse
import os , os.path as osp
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader , Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils import (mkdir_or_exist  , update_config , setup_seed , init_dist)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def args_parse():
    parser = argparse.ArgumentParser(description="OroModelZoo Segmentation")
    # Model training settings
    parser.add_argument("--config",type=str,required=True,
                        help="Train experiment configure file path")
    # General settings
    parser.add_argument("--nThreads",type=int,default=60,
                        help="Number of data loading threads")
    parser.add_argument("--seed",type=int,default=2000,
                        help="random seed")
    # Distributed Data Parallel settings
    parser.add_argument("--rank",type=int,default=-1,
                            help="Node rank ofr distributed training")
    parser.add_argument("--address",type=str,default="127.0.0.1:",
                            help="Url used to set up distributed training")
    parser.add_argument("--port",type=str,default="23456",
                            help="Port used to set up distributed training")
    parser.add_argument("--backend", type=str, default='nccl', help="distributed backend")
    parser.add_argument("--launcher", choices=['none', 'pytorch','slurm','mpi'],
                            default='pytorch', help='job launcher')

    args = parser.parse_args()
    return args


def main():
    # Build config from yaml file
    args = args_parse()
    if args.config is not None:
        assert args.config.endswith('.yaml') , \
            f"Config file should be a yaml , Instead of {osp.splitext(args.config)[-1]}"
    cfg_file_name = osp.basename(args.config)
    cfg = update_config(args.config)
    print("# cfg # : " , cfg)

    cfg['FILE_NAME'] = cfg_file_name
    args.world_size = cfg.TRAIN.world_size
    num_gpu = torch.cuda.device_count()
    if args.world_size > num_gpu:
        print(f"The config of world_size does not match the available number of devices. "
              f"Changing it from {args.world_size} to {num_gpu}")
        args.world_size = num_gpu

    # Build WorkDir to save checkpoints and log
    if cfg.WORK_DIR is None :
        args.work_dir = './exp/{}-{}/'.format(osp.splitext(cfg_file_name)[0] , time.strftime("%Y%m%d-%H%M"))
    # TODO
    # mkdir_or_exist(args.work_dir)
    args.gpus = [i for i in range(torch.cuda.device_count())]
    args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")

    if not cfg.MODEL.dynamic :
        # set cudnn_benchmark to accelerate the network.
        # The applicable scenario is that the network structure is fixed (not dynamic)
        torch.backends.cudnn.benchmark = True

    if args.seed is not None:
        setup_seed(args.seed)

    # TODO
    if args.launcher == 'slurm':
        # Use the slurm computing cluster for training
        main_worker(None , args , cfg)
    else :
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        # The first parameter of mp.spawn is a function, this function will execute all the steps of training, python will create multiple processes, each process will execute the main_worker function.
        # The second parameter is the number of processes to open.
        # The third parameter is the function argument of main_worker
        mp.spawn(main_worker , nprocs = ngpus_per_node , args=(args , cfg))


def main_worker(gpu , args , cfg):
    if gpu is not None :
        args.gpu = gpu

    init_dist(args)
    exit(0)


    # init logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')


    # build datasets

    # SyncBN is not support in DP mode
    if distributed_:
        train_sampler = DistributedSampler(dataset="", shuffle=True)
        val_sampler = DistributedSampler(dataset="", shuffle=False)

    train_dataloader = DataLoader("", batch_size=args.batch_size)

    # Init Automatic mixed precision GradScaler
    if args.fp16:
        # torch.cuda.amp provides users with a more convenient mixed-precision training mechanism.
        # Users do not need to manually convert the model parameter dtype, amp will automatically select the appropriate numerical precision for the operator
        # For the problem of FP16 gradient value overflow during backpropagation, amp provides a gradient scaling operation,
        # and before the optimizer updates the parameters, it will automatically unscale the gradient, so there will be no hyperparameter for model optimization.
        from torch.cuda.amp import GradScaler as GradScaler
        scaler_ = GradScaler()
    else:
        scaler_ = None

    # Init tensorboard summarywriter
    writer_ = SummaryWriter('./logs')

    model_ = None

    if args.checkpoint is None:
        weights_init(model_)
    else:
        # Load according to the Key of the pre-trained weight and the Key of the model
        model_dict_ = model_.state_dict()
        pretrain_dict_ = torch.load(args.checkpoint, map_location=args.device)
        success_load, fail_load, temp_dict = [], [], {}
        for k, v in pretrain_dict_.items():
            if k in model_dict_.keys() and np.shape(model_dict_[k]) == np.shape(v):
                temp_dict[k] = v
                success_load.append(k)
            else:
                fail_load.append(k)
        model_dict_.update(temp_dict)
        model_.load_state_dict(model_dict_)
        if rank == 0:
            print("Successfully Load Key : ", str(success_load))
            print("Fail Load Key : ", str(fail_load))

    if distributed_:
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_)

    # loss function
    criterion_ = nn.CrossEntropyLoss()
    # optimizer
    if args.TRAIN.optimizer == 'Adam':
        optimizer_ = optim.Adam(model_.parameters(), lr=args.TRAIN.LR)
    elif args.TRAIN.optimizer == 'SGD':
        optimizer_ = optim.SGD(model_.parameters(), lr=args.TRAIN.LR, momentum=0.9, weight_decay=0.0001)
    if args.TRAIN.multistep_lr:
        lr_scheduler_ = optim.lr_scheduler.MultiStepLR(optimizer=optimizer_, milestones=args.TRAIN.LR_STEP,
                                                       gamma=args.TRAIN.LR_FACTOR)

    metric_ = None

    for epoch in range(args.start_epoches, args.num_epoches):
        model_.train()
        optimizer_.zero_grad()
        loss_ = criterion_()
        if scaler_ is not None:
            scaler_.scale(loss=loss_).backward()
            scaler_.step(optimizer=optimizer_)
            scaler_.update()
        else:
            loss_.backward()
            optimizer_.step()
        writer_.add_scalar("loss function", loss_.item(), epoch)

        # Save model last ckpt , best ckpt and delete the value ckpt
        torch.save()

    writer_.close()
    pass

if __name__ == '__main__':
    main()
    