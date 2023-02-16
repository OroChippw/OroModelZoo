import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset

from ..builder import DATASETS
from ..transforms import Compose

@DATASETS.register_module()
class BaseDataset(Dataset):
    '''
    Description:
        BaseDataset is the root dataset for semantic segmentation.
        The image/ground_truth pair of BaseDataset should be of the same except shuffix.
        The contents of train_path/val_path should as follow,the separator of dataset list. Default: ' ':
            image_1.jpg image_1_ground_truth.png
            image_2.jpg image_2_ground_truth.png
    Args:
        dataset_root(str): The dataset directory
        transforms(list): Transforms for image
        num_classes(int): Number of classes
        mode(str): which part dataset to use
        train_path(str): The train dataset filepath
        val_path(str): The val dataset filepath
        test_path(str): The test dataset filepath
        
    '''
    def __init__(self , dataset_root , transforms ,num_classes , mode , img_channels=3  , 
                train_path=None , val_path=None , test_path=None , ignore_index=255):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms , img_channels=img_channels)
        self.num_classes = num_classes
        self.mode = mode.lower()
        self.file_list = list()
        
        if not osp.exists(self.dataset_root):
            raise FileNotFoundError(f'{self.dataset_root} is not exists')
        if img_channels not in [1,3]:
            raise ValueError(f'`img_channels` should in [1,3] , but got {self.img_channels}')
        if self.num_classes < 1:
            raise ValueError(f'num_classes should at least 1 , but got {self.num_classes}')
        if self.mode not in ['train' , 'val' , 'test']:
            raise ValueError(f'mode should be `train , `val` , `test` , but got {self.mode}')
        if self.mode == 'train':
            if train_path is None:
                raise ValueError('When mode is `train`,train_path is necessary,but got None')
            elif not osp.exist(train_path):
                raise FileNotFoundError(f'{train_path} is not exist')
            else :
                file_path = train_path
        elif self.mode == 'val':
            if val_path is None:
                raise ValueError('When mode is `val`,val_path is necessary,but got None')
            elif not osp.exist(val_path):
                raise FileNotFoundError(f'{val_path} is not exist')
            else :
                file_path = val_path
        else :
            if test_path is None:
                raise ValueError('When mode is `test`,test_path is necessary,but got None')
            elif not osp.exist(test_path):
                raise FileNotFoundError(f'{test_path} is not exist')
            else :
                file_path = test_path
        
        with open(osp.abspath(file_path) , 'r') as f:
            for line_ in f:
                item_ = line_.strip().split(' ')
                if len(item_) != 2:
                    if self.mode in ['train' , 'val']:
                        raise ValueError('File list format incorrect')
                    if not osp.isabs(item_[0]):
                        image_path = osp.join(self.dataset_root , item_[0])
                        label_path = None
                else :
                    if not osp.isabs(item_[0]):
                        image_path = osp.join(self.dataset_root , item_[0])
                        label_path = osp.join(self.dataset_root , item_[1])
                    else:
                        image_path = item_[0]
                        label_path = item_[1]
                self.file_list.append([image_path , label_path])
    
    
    def __getitem__(self , idx):
        data = {}
        image_path , label_path = self.file_list[idx]
        data['image'] = image_path
        data['label'] = label_path
        if self.mode == 'val':
            data = self.transforms(data)
            data['label'] = data['label'][np.newaxis , : , :]
        else :
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.file_list)
        