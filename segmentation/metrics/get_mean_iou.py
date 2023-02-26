import numpy as np

# ----------------------------------- #
# 构建混淆矩阵根据对角线计算mIoU #
# ----------------------------------- #
def build_hist(a , b , n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 


# ----------------------------------- #
# -----直接输入区域进行计算的mIoU----- #
# ----------------------------------- #
def cal_mean_iou(intersect_area , pred_area , label_area):
    '''
        Calculate IoU
    '''
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
            class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou) , miou