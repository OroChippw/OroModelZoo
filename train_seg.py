import argparse
import os , os.path as osp
import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader , Dataset
from torch.utils.data.distributed import DistributedSampler

def args_parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--checkpoint",type=str,default="",help="")
    parser.add_argument("--epoches",type=int,default=2000,help="")
    parser.add_argument("--batch-size","-b",type=int,default=64,help="")
    parser.add_argument("--launcher",choices=['none','pytorch'],default='pytorch',help='job launcher')
    parser.add_argument("--local-rank",type=int,default=-1,help="")
    parser.add_argument("--backend",type=str,default='nccl',help="")
    parser.add_argument("--des-dir",type=str,default="",help="")
    parser.add_argument("--num_worker",type=int,default=4,help="")
    args = parser.parse_args()
    return args

def main():
    args = args_parse()
    local_rank_ = args.local_rank
    backend_ = args.backend
    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batchsize_ = args.batch_size

    # set cudnn_benchmark to accelerate the network. 
    # The applicable scenario is that the network structure is fixed (not dynamic)
    torch.backends.cudnn.benchmark = True

    if args.des_dir is not None :
        pass


    torch.cuda.set_device(local_rank_)
    dist.init_process_group(backend=backend_)

    if args.launcher == 'none':
        distributed_ = False
    else :
        distributed_ = True

    train_sampler = DistributedSampler(dataset="")

    train_dataloader = DataLoader("" , batch_size=batchsize_)




if __name__ == '__main__':
    main()
    