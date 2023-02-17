import random
from ..builder import TRANSFORMS

@TRANSFORMS.register_module()
class RandomNoise():
    '''
    Func:
         Superimposing noise on an image with a certain probability.
    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.5.
        max_sigma(float, optional): The maximum value of standard deviation of the distribution.(分布标准差的最大值)
            Default: 10.0.
    '''
    def __init__(self , prob=0.5 , max_sigma=10.0) -> None:
        self.prob = prob
        self.max_sigma = max_sigma
    
    def __call__(self, data):
        
        pass