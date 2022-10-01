import os.path as osp
from re import S
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self , data_root : str = None , ann_info : str = None ,
                classes  = None) -> None:
        super(BaseDataset).__init__()
        self.data_root = data_root
        self.ann_info = ann_info
        self.classes = self._get_classes(classes)

        if self.data_root is not None :
            if not osp.isabs(self.ann_info):
                self.ann_info = osp.join(self.data_root , self.ann_info)



    def _get_classes(self , classes = None):
        classes_list = []
        if classes is None : 
            return None
        elif isinstance(classes , (tuple , list)):
            classes_list = classes
        elif isinstance(classes ,str):
            pass
        else :
            raise ValueError(f'Unsupported type {type(classes)} of classes.')