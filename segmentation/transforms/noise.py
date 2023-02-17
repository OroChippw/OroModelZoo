import cv2
import random
import numpy as np

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
        if random.random() > self.prob:
            sigma = random.random() * self.max_sigma
            mu = 0
            data["image"] = np.array(data["image"] ,dtype=np.float32)
            data["image"] += np.random.normal(mu , sigma , data["image"].shape)
            data["image"][data["image"] > 255] = 255
            data["image"][data["image"] < 0] = 0
        
        return data
    
@TRANSFORMS.register_module()    
class RandomBlur():
    '''
    Func:
        Blurring an image by a Gaussian function with a certain probability.
    Args:
        prob (float, optional): A probability of blurring an image. Default: 0.1.
        blur_type(str, optional): A type of blurring an image,
            gaussian stands for cv2.GaussianBlur,
            median stands for cv2.medianBlur,
            blur stands for cv2.blur,
            Default: gaussian.
    '''
    def __init__(self , prob=0.1 , blur_type="gaussian"):
        self.prob = prob
        self.blur_type = blur_type
    
    def __call__(self, data):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 /self.prob)
        
        if n > 0:
            if np.random.randint(0,n) == 0:
                radius = np.random.randint(3,10)
                if radius % 2 != 1:
                    radius += 1
                if radius > 9:
                    radius = 9
                data["image"] = np.array(data["image"] ,dtype='uint8')
                if self.blur_type == "gaussian":
                    data["image"] = cv2.GaussianBlur(data["image"] , (radius , radius) , 0 , 0)
                elif self.blur_type == "median":
                    data["image"] = cv2.medianBlur(data["image"] , (radius , radius))
                elif self.blur_type == "blur":
                    data["image"] = cv2.blur(data["image"] , (radius , radius))
        data["image"] = np.array(data["image"] , dtype='float32')
        return data
                    
                    
                