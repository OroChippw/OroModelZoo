import cv2
from ..builder import TRANSFORMS

INTERPOLATION_MODE = {
    # https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.7/paddleseg/transforms/transforms.py
}

@TRANSFORMS.register_module()
class Resize():
    def __init__(self) -> None:
        pass