import logging

import torch.distributed as dist

init_logger_list : dict = {}

def logger_maker(logger_name , log_file , log_level=logging.INFO):
    logger = get_logger(logger_name=logger_name , log_file=log_file , log_level=log_level)
    return logger

def get_logger(logger_name , log_file , log_level =logging.INFO , file_mode='w'):
    logger = logging.getLogger(logger_name)
    if logger_name in init_logger_list:
        return logger
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else :
        rank = 0
    
    if rank == 0 :
        logger.setLevel(log_level)
    else :
        logger.setLevel(logging.ERROR)
    
    init_logger_list[logger_name] = True

    return logger

def print_logger(msg , logger , level = logging.INFO):
    if logger is None:
        print(msg)
    elif isinstance(logger , logging.Logger):
        logger.log(level , msg)
    else :
        raise TypeError(
            'logger should be a logging.Logger'
        )
    


