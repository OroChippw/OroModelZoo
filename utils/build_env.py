import os 
import platform
import cv2 

import torch.multiprocessing as mp

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

    