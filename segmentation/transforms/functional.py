import cv2
import numpy as np

def _cal_rescale_size(image_size , target_size):
    '''
    example:
        image_size : 1920 * 1080 ; target_size : 512 * 512 
        scale = min((512 / 1920 = 0.266) , (512 / 1080 = 0.474)) = 0.266
        rescaled_size = [1920 * 0.266 , 1080 * 0.266] = [512 , 288]    
    '''
    scale = min(
        max(target_size) / max(image_size) , 
        min(target_size) / min(image_size)
    )
    rescaled_size = [round(i * scale) for i in image_size]
    return rescaled_size , scale

def _resize(image , target_size=512 , interp=cv2.INTER_LINEAR):
    if isinstance(target_size , list) or isinstance(target_size , tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    image = cv2.resize(image , (w ,h) , interpolation=interp)
    return image

def _horizontal_flip(image):
    if len(image.shape) == 3:
        image = image[: , ::-1 , :]
    elif len(image.shape) == 2:
        image = image[: , ::-1]
    return image

def _vertical_flip(image):
    if len(image.shape) == 3:
        image = image[::-1 , : , :]
    elif len(image.shape) == 2:
        image = image[::-1 , :]
    return image

def _normalize(image , mean , std):
    image = image.astype(np.float32 , copy=False) / 255.0
    image -= mean
    image /= std
    return image