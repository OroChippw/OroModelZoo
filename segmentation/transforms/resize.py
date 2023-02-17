import cv2
import math
from ..builder import TRANSFORMS
from .functional import _cal_rescale_size , _resize

INTERPOLATION_MODE = {
    'NEAREST': cv2.INTER_NEAREST,
    'LINEAR': cv2.INTER_LINEAR,
    'CUBIC': cv2.INTER_CUBIC,
    'AREA': cv2.INTER_AREA,
    'LANCZOS4': cv2.INTER_LANCZOS4
}

@TRANSFORMS.register_module()
class Resize():
    """
    Func:
        Resize an image
    Args:
        target_size (list|tuple, optional): The target size (w, h) of image. Default: (192, 192)
        keep_ratio (bool, optional): Whether to keep the same ratio for width and height in resizing.
            Default: False.
        size_divisor (int, optional): If size_divisor is not None, make the width and height be the times
            of size_divisor. Default: None.
        interp (str, optional): The interpolation mode of resize is consistent with opencv.
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'].
    """
    def __init__(self , target_size=(192,192) , 
                 keep_ratio=False , size_divisor=None , interp='LINEAR') -> None:
        if isinstance(target_size , list) or isinstance(target_size , tuple):
            if len(target_size) != 2:
                raise ValueError(f"`target_size` should include 2 elements , but it is {type(target_size)}")
        else:
            raise ValueError(f"`target_size` should be list or tuple ,but it is {type(target_size)}")
        if size_divisor is not None:
            assert isinstance(size_divisor , int) , "size_divisor should be None or int"
        
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.size_divisor = size_divisor
        self.interp = interp
    
    def __call__(self, data):
        data['trans_info'].append(('resize' , data['image'].shape[0:2]))
        interp_ = self.interp
        target_size = self.target_size
        if self.keep_ratio:
            h ,w = data['image'].shape[0:2]
            target_size = _cal_rescale_size((w , h) , self.target_size)
        if self.size_divisor:
            target_size = [
                math.ceil(i / self.size_divisor) * self.size_divisor for i in target_size
            ]
            
        data["image"] = _resize(data['image'] , target_size , INTERPOLATION_MODE[interp_])
        
        for key in data.get('gt_fields' , []):
            data[key] = _resize(data[key] , target_size , INTERPOLATION_MODE[interp_])
        
        return data
            
            
            
            
        