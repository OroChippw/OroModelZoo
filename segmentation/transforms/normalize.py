import numpy as np
# Use the function iteratively on a sequence to reduce the incoming sequence to a value.
from functools import reduce

from ..builder import TRANSFORMS
from .functional import _normalize

@TRANSFORMS.register_module()
class Normalize():
    '''
    Func:
        Normalize an image
    Args:
        mean (list, optional): The mean value of a data set. Default: [0.5,].
        std (list, optional): The standard deviation(标准偏差) of a data set. Default: [0.5,].
    '''
    def __init__(self , mean=(0.5,) , std=(0.5,)) -> None:
        if not (isinstance(mean , (list , tuple)) and isinstance(std , (list , tuple))) \
            and (len(mean) not in [1,3]) and (len(std) not in [1,3]):
                raise ValueError(f"mean or std input type is invalid .It should be list or tuple with the length of 1 or 3")
        self.mean = np.array(mean)
        self.std = np.array(std)
        
        if reduce(lambda x , y : x * y , self.std) == 0:
            raise ValueError(f"std is invalid")
        
    def __call__(self, data):
        data["image"] = _normalize(data["image"] , self.mean , self.std)
        return data
         