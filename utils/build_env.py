import os 
import platform
import numpy as np
import random
import cv2
import torch

import torch.multiprocessing as mp
import torch.distributed as dist

def build_multi_process(mp_method = None):
    if platform.system() != 'Windows':
        mp_method = 'fork'
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_method:
            pass
        mp.set_start_method(mp_method , force = None)

    # disable opencv multithreading to avoid the problem of interlocking between OpenCV and Pytorch
    opencv_num_threads = 0
    cv2.setNumThreads(opencv_num_threads)

def setup_seed(seed = 2000):
    random.seed(seed)
    '''
        os.environ['PYTHONHASHSEED'] = str(seed) : Disable hash randomization
        Ensure reproducible results run-to-run to determine whether changes in performance are 
        due to model or dataset changes, or just a few new random sample points
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # for CPU
    torch.manual_seed(seed)
    '''
        torch.cuda.manual_seed(seed) : for GPU
        Sets the seed for generating random numbers. 
        When the set seed is fixed, the sequence of random numbers generated by pytorch is also fixed.
    '''
    torch.cuda.manual_seed(seed)
    '''
        torch.cuda.manual_seed_all(seed) : for GPU
        Set seeds for all GPUs to generate the same sequence of random numbers
    '''
    torch.cuda.manual_seed_all(seed)
    '''
        benchmark is set to False, and deterministic is set to True to ensure the use of a 
        deterministic convolution algorithm to ensure the consistency of convolution operations
        but it will sacrifice part of the training speed
    '''
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def init_dist(args):
    args.ngpus_per_node = torch.cuda.device_count()
    launcher_ = args.launcher
    if launcher_ == 'pytorch':
        _init_dist_pytorch(args)
        return True
    elif launcher_ == 'slurm' :
        _init_dist_slurm(args)
    elif launcher_ == 'mpi' :
        _init_dist_mpi(args)
    elif launcher_ == 'none' :
        args.world_size = 1
        args.rank = 0
        args.gpu = 0
        args.log = True
        dist.init_process_group(backend=args.backend , init_method=args.dist_url , 
                                    world_size=args.world_size , rank=args.rank)
        torch.cuda.set_device(args.gpu)
        return False
    else :
        raise ValueError(f"Launcher type shoud be in [/'none/', 'pytorch','slurm','mpi'] , ' \
                            'Unsupport launcher type : {launcher_}")


# TODO
def _init_dist_pytorch(args , **kwargs):
    # print("args.rank" , args.rank)
    # print("args.ngpus_per_node" , args.ngpus_per_node)
    # print("args.gpu" , args.gpu)
    args.rank = args.rank * args.ngpus_per_node + args.gpu
    num_gpus = torch.cuda.device_count()
    print(f"dist_url : {args.dist_url} , world_size : {args.world_size} , rank : {args.rank}")
    if platform.system() != 'Windows':
        dist.init_process_group(backend=args.backend , init_method=args.dist_url ,
                                world_size=args.world_size , rank=args.rank)
    else :
        # Windows system unsupport NCCL backend
        dist.init_process_group(backend="gloo" , init_method="file:///sharefile" ,
                                world_size=args.world_size , rank=args.rank)
    torch.cuda.set_device(args.gpu)
    if args.rank % args.ngpus_per_node == 0 :
        print("args.log")
        args.log = True
    else :
        args.log = False

# TODO
def _init_dist_slurm(args , **kwargs):
    raise NotImplementedError

# TODO
def _init_dist_mpi(args , **kwargs):
    raise NotImplementedError

    