
import random

from ..builder import TRANSFORMS
from .functional import _horizontal_flip , _vertical_flip

@TRANSFORMS.register_module()
class RandomHorizontalFlip():
    '''
    Func:
        Flip an image horizontally with a certain probability.
    Args:
        prob (float, optional): A probability of horizontally flipping. Default: 0.5.
    '''
    def __init__(self , prob=0.5) -> None:
        self.prob = prob
    
    def __call__(self, data):
        # The random() method returns a randomly generated real number in the range [0,1).
        if random.random() > self.prob:
            data["image"] = _horizontal_flip(data["image"])
            for key in data.get('gt_fields' , []):
                data[key] = _horizontal_flip(data[key])
            
        return data
    
@TRANSFORMS.register_module()
class RandomVerticalFlip():
    '''
    Func:
        Flip an image vertically with a certain probability.
    Args:
        prob (float, optional): A probability of vertical flipping. Default: 0.5.
    '''
    def __init__(self , prob=0.5) -> None:
        self.prob = prob
    
    def __call__(self, data):
        # The random() method returns a randomly generated real number in the range [0,1).
        if random.random() > self.prob:
            data["image"] = _vertical_flip(data["image"])
            for key in data.get('gt_fields' , []):
                data[key] = _vertical_flip(data[key])
            
        return data