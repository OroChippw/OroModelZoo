import os.path as osp

from .base import BaseDataset

@DATASETS.register_module()
class PPHumanSeg14K(BaseDataset):
    def __init__(self , dataset_root , transforms , mode):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.mode = mode.lower()
        self.num_classes = 2
        self.file_list = list()
        
        
        if self.mode is not in ['train' , 'val' , 'test']:
            raise ValueError(f'mode should be `train , `val` , `test` , but got {self.mode}')
        if self.mode == 'train':
            file_path = osp.join(self.dataset_root , 'train.txt')
        elif self.mode == 'val':
            file_path = osp.join(self.dataset_root , 'val.txt')
        else :
            file_path = osp.join(self.dataset_root , 'test.txt')
        
        with open(file_path , 'r') as f:
            for line_ in f:
                item_ = line_.strip().split(' ')
                if len(item_) != 2:
                    if self.mode in ['train' , 'val']:
                        raise ValueError('File list format incorrect')
                    image_path = osp.join(self.dataset_root , item_[0])
                    label_path = None
                else :
                    image_path = osp.join(self.dataset_root , item_[0])
                    label_path = osp.join(self.dataset_root , item_[1])
                self.file_list.append([image_path , label_path])
        