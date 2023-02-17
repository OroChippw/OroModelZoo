import cv2
import numpy as np

from ..builder import TRANSFORMS

@TRANSFORMS.register_module()
class RandomRotaion():
    '''
    Func:
        Rotate an image randomly with padding.
    Args:
        max_rotation (float, optional): The maximum rotation degree. Default: 15.
        im_padding_value (float, optional): The padding value of raw image. Default: 127.5.(图像的填充值)
        label_padding_value (int, optional): The padding value of annotation image. Default: 255.(标签的填充值)
    Principle:
        https://cloud.tencent.com/developer/article/1798209
    '''
    def __init__(self , max_rotation=15 ,
                 im_padding_value=127.5,
                 label_padding_value=255) -> None:
        self.max_rotation = max_rotation
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value
        
        
    def __call__(self, data):
        if self.max_rotation > 0:
            (h , w) = data["image"].shape[:2]
            img_channels = 1 if data['image'].ndim == 2 else data['image'].shape[2]
            do_rotation = np.random.uniform(-self.max_rotation , self.max_rotation)
            center = (w // 2 , h // 2)
            '''
            M = cv2.getRotationMatrix2D(center, angle, scale)
            parameters:
                center:旋转中心坐标，是一个元组参数(col, row)
                angle:旋转角度，旋转方向，负号为顺时针，正号为逆时针
                scale:旋转后图像相比原来的缩放比例，1为等比例缩放
            return:
                M:一个2*3的旋转矩阵M
            '''
            rotate_matrix = cv2.getRotationMatrix2D(center , do_rotation , 1.0)
            cos = np.abs(rotate_matrix[0,0])
            sin = np.abs(rotate_matrix[0,1])
            
            # When the size of the canvas remains the same, part of the image will exceed and the display will not be complete, 
            # so we need to expand the canvas
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            (cx , cy) = center
            rotate_matrix[0 , 2] += (new_w / 2) - cx
            rotate_matrix[1 , 2] += (new_h / 2) - cy
            dsize = (new_w , new_h)
            """
            cv2.warpAffine(src, M, dsize, dst, flags, borderMode, borderValue) → dst
                cv2.warpAffine()函数主要是利用变换矩阵M对图像进行如旋转、仿射、平移等变换，
            只需要我们提供一个2*3的变换矩阵M，就可以对图像进行变换。它一般是和cv2.getRotationMatrix2D和
            cv.GetAffineTransform两个函数在一起使用，这两个函数是用来获取变换矩阵M,这样就不需要我们自己设置M。
            parameters:
                src:输入图像，可以为单通道或彩色图像
                M:变换矩阵，一般反映平移或旋转的关系，为InputArray类型的2×3变换矩阵
                dsize:输出图像的大小
                flags:插值方法的组合（int 类型），默认为 flags=cv2.INTER_LINEAR，表示线性插值
                borderMode:边界像素模式（int 类型）
                borderValue:边界填充值; 默认情况下，它为0，也就是边界填充默认是黑色。
            return:
                dst: 仿射变换后的输出图像
            """
            data["image"] = cv2.warpAffine(data["image"] , rotate_matrix , dsize=dsize , 
                                           flags=cv2.INTER_LINEAR , borderMode=cv2.BORDER_CONSTANT , 
                                           borderValue=(self.im_padding_value , ) * img_channels)
            for key in data.get('gt_fields' , []):
                data[key] = cv2.warpAffine(
                    data[key] , rotate_matrix , dsize=dsize , flags=cv2.INTER_LINEAR , 
                    borderMode=cv2.BORDER_CONSTANT , borderValue=self.label_padding_value
                )
        
        return data
            
            
            
            
            
        return data