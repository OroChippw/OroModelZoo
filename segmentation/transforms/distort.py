import random
import numpy as np
from PIL import Image

from ..builder import TRANSFORMS
from .functional import _brightness , _contrast , _hue , _sharpness , _saturation

@TRANSFORMS.register_module()
class RandomDistort():
    '''
    Func:
        Distort an image with random configurations.
    Args:
        Args:
        brightness_range (float, optional): A range of brightness. Default: 0.5.(亮度范围)
        brightness_prob (float, optional): A probability of adjusting brightness. Default: 0.5.(调整亮度概率)
        contrast_range (float, optional): A range of contrast. Default: 0.5.(对比度范围)
        contrast_prob (float, optional): A probability of adjusting contrast. Default: 0.5.(调整对比度概率)
        saturation_range (float, optional): A range of saturation. Default: 0.5.(饱和度范围)
        saturation_prob (float, optional): A probability of adjusting saturation. Default: 0.5.(调整饱和度概率)
        hue_range (int, optional): A range of hue. Default: 18.(色调范围)
        hue_prob (float, optional): A probability of adjusting hue. Default: 0.5.(调整色调的概率)
        sharpness_range (float, optional): A range of sharpness. Default: 0.5.(锐度范围)
        sharpness_prob (float, optional): A probability of adjusting saturation. Default: 0.(调整锐度的概率)
    '''
    def __init__(self , brightness_range=0.5 , brightness_prob=0.5,
                 contrast_range=0.5 , contrast_prob=0.5,
                 saturation_range=0.5 , saturation_prob=0.5,
                 hue_range=18 , hue_prob=0.5,
                 sharpness_range=0.5 , sharpness_prob=0):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob
        self.sharpness_range = sharpness_range
        self.sharpness_prob = sharpness_prob
        
    def __call__(self, data):
        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        sharpness_lower = 1 - self.sharpness_range
        sharpness_upper = 1 + self.sharpness_range
        
        ops = [
            _brightness , _contrast , _saturation , _sharpness
        ]
        
        if data["image"].ndim > 2:
            ops.append(_hue)
            
        random.shuffle(ops)
        
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            },
            'sharpness': {
                'sharpness_lower': sharpness_lower,
                'sharpness_upper': sharpness_upper,
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
            'sharpness': self.sharpness_prob
        }
        
        data["image"] = data["image"].astype('uint8')
        data["image"] = Image.fromarray(data["image"])
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['image'] = data['image']
            if np.random.uniform(0,1) > prob:
                data["image"] = ops[id](**params)
        data["image"] = np.array(data["image"]).astype("float32")
        
        return data
    