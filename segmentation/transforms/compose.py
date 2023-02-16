import cv2

from ..builder import TRANSFORMS


@TRANSFORMS.register_module()
class Compose():
    def __init__(self , transforms , img_channels) -> None:
        self.transforms = transforms
        self.img_channels = img_channels
        self.read_flag = cv2.IMREAD_GRAYSCALE if self.img_channels == 1 else cv2.IMREAD_COLOR
        
    def __call__(self, data):
        if 'image' not in data.keys():
            raise ValueError("data must include `image` key")
        if isinstance(data['image'] , str):
            data['image'] = cv2.imread(data['image'] , self.read_flag).astype('float32')
            