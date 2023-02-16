import cv2
import imghdr
import numpy as np
from PIL import Image

from ..builder import TRANSFORMS


@TRANSFORMS.register_module()
class Compose():
    def __init__(self , transforms , to_rgb=True , img_channels=3) -> None:
        self.transforms = transforms
        self.to_rgb = to_rgb
        self.img_channels = img_channels
        self.read_flag = cv2.IMREAD_GRAYSCALE if self.img_channels == 1 \
                            else cv2.IMREAD_COLOR
        
    def __call__(self, data):
        if 'image' not in data.keys():
            raise ValueError("data must include `image` key")
        assert imghdr.what(data['image']) is not None , f"Image {data['image']} is corrupted"
        if isinstance(data['image'] , str):
            data['image'] = cv2.imread(data['image'] , self.read_flag).astype('float32')
        if not isinstance(data['img'], np.ndarray):
            raise TypeError("Image type is not numpy.")
        if self.to_rgb and self.img_channels == 3:
            data['image'] = cv2.cvtColor(data['image'] , cv2.COLOR_BGR2RGB)
        if 'label' in data.keys() and isinstance(data['label'] , str):
            data['label'] = np.asarray(Image.open(data['label']))
        if 'trans_info' not in data.keys():
            data['trans_info'] = []
            
        for t in self.transforms:
            data = t(data)
        
        if data['image'].ndim == 2:
            data['image'] = data['image'][... , np.newaxis]
        data['image'] = np.transpose(data['image'] , (2,0,1))
        return data
            
            
         
        