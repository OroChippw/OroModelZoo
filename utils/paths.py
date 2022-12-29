import os , os.path as osp
from pathlib import Path

def is_string(x):
    # whether the input is an string instance
    return isinstance(x , str)

def mkdir_or_exist(dir_name , mode=0o777):
    if dir_name == '':
        return
    os.makedirs(dir_name , mode=mode , exist_ok=True)
    

