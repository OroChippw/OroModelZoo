
import os.path as osp

from .base import BaseDataset
from ..builder import DATASETS
from .transforms import Compose


@DATASETS.register_module()
class PascalVOC(BaseDataset):
    def __init__(self , dataset_root , transforms ,  mode ):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.mode = mode.lower()
        self.num_classes = 21
        self.file_list = list()
        
        if mode not in ['train' , 'trainval' , 'trainaug' , 'val']:
            raise ValueError(f'mode should be `train , `val` , `trainval` , `trainaug`, but got {self.mode}')

        image_set_dir = osp.join(self.dataset_root , 'ImageSets' , 'Segmentation')
        if self.mode == 'train':
            file_path = osp.join(image_set_dir, 'train.txt')
        elif self.mode == 'val':
            file_path = osp.join(image_set_dir, 'val.txt')
        elif self.mode == 'trainval':
            file_path = osp.join(image_set_dir, 'trainval.txt')
        elif self.mode == 'trainaug':
            file_path = osp.join(image_set_dir, 'train.txt')
            file_path_aug = os.path.join(image_set_dir, 'aug.txt')
            if not osp.exists(file_path_aug):
                raise RuntimeError(
                    "When `mode` is 'trainaug', Pascal Voc dataset should be augmented, "
                    "Please make sure voc_augment.py has been properly run when using this mode."
                )
        
        image_dir = os.path.join(self.dataset_root, 'VOC2012', 'JPEGImages')
        label_dir = os.path.join(self.dataset_root, 'VOC2012', 'SegmentationClass')
        label_dir_aug = os.path.join(self.dataset_root, 'VOC2012', 'SegmentationClassAug')
        
        with open(file_path , 'r') as f:
            for line_ in f:
                line_ = line_.strip()
                image_path = osp.join(image_dir , ''.join([line_ , '.jpg']))
                label_path = osp.join(label_dir , ''.join([line_ , '.png']))
                self.file_list.append([image_path , label_path])
        if self.mode == 'trainaug':
            with open(file_path_aug , 'r') as f:
                for line_ in f:
                    line_ = line_.strip()
                    image_path = osp.join(image_dir , ''.join([line_ , '.jpg']))
                    label_path = osp.join(label_dir_aug , ''.join([line_ , '.png']))
                    self.file_list.append([image_path , label_path])
            
                
                
    

    