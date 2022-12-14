from .alexnet import AlexNet
from .mobilenet_v1 import MobileNetv1
from .mobilenet_v2 import MobileNetv2
from .mobilenet_v3 import MobileNet_v3_small , MobileNet_v3_large
from .shufflenet_v1 import ShuffleNetv1
from .shufflenet_v2 import ShuffleNetv2
from .unet import UNet
from .cgnet import CGNet

__all__ = ["AlexNet" , "MobileNetv1" , "MobileNetv2" , "MobileNet_v3_small" , "MobileNet_v3_large" , 
            "ShuffleNetv1" , "ShuffleNetv2" , "UNet" , "CGNet"]