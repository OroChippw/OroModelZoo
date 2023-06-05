from typing import Any
from PIL import Image

import torch
from torch.utils.data import Dataset

class SampleDataSet(Dataset):
    def __init__(self , image_path , image_classes , transform=None) -> None:
        super(SampleDataSet , None).__init__()
        self.image_path = image_path
        self.class_list = image_classes
        self.transform = transform
    
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, item) -> Any:
        img = Image.open(self.image_path[item])
        assert img.mode == 'RGB' , \
            ValueError(f"image {self.image_path[item]} isn't RGB mode.")

        label = self.class_list[item]
        if self.transform is not None:
            img = self.transform(img)
            
        return img , label
    
    def collate_fn(batch):
        images , labels = tuple(zip(*batch))
        images = torch.stack(images , dim=0)
        labels = torch.as_tensor(labels)
        return images , labels