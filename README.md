# OroModelZoo

## Introduction
  OroModelZoo is a PyTorch-based open source toolbox for deep learning network reproduction involving classification, segmentation, and detection of computer vision directions.  
  At present, it is mainly used for personal study and experiment of OroChi Fang.Welcome everyone to submit and provide constructive suggestions for network reproduction, thank you🤞
## Top News
**`2023-02`**:  
**`2022-10`**: **Create repository for OroModelZoo🎂**

## Supported model and backbone
### backbone
- [x] MobileNetv1
+ MobileNet is a lightweight deep nerural network proposed by Google for embedded devices such as mobile phones. 
+ Proposed Depthwise Separable Convolution,3 * 3 Depthwise Conv + 1 * 1 Pointwise Conv(深度可分离卷积).  
+ Use ReLU6 install of ReLU,The activation value can be distributed in a small range, and low-precision float16 and other embedded devices can be accurately described, thereby avoiding loss of precision.
+ Limitation:(1)No residual connections(2)Because the number of convolution kernel weights is too small,many Depthwise convolution kernels are trained to be 0.
- [x] MobileNetv2
+ Inverted Residual Block(倒残差结构)
+ Linear Bottleneck(线性瓶颈)
+ Limitation: 
- [x] MobileNetv3  
    1.更新了Block提出bneck  
    2.加入SE注意力机制，更新激活函数，  
    swish[x] 计算求导复杂不好量化
    h-sigmoid 
    h-swish[x]  
    3.Neural Architecture Search 搜索参数  
    重构耗时结构  
    Modify the number of channels of the head convolution kernel. Mobilenet v2 uses 32 x 3 x 3. The author found that 32 can actually be reduced a little bit, so here the author changed it to 16, which is reduced by 3ms on the premise of ensuring the accuracy. speed
- [ ] ShuffleNetV1
- [ ] ShuffleNetV2
### Segmentation
- [x] UNet
+ One of the earlier algorithms for semantic segmentation tasks using multi-scale features(较早使用多尺度特征进行语义分割任务的算法之一)
+ The fully convolutional neural network (FCN) is introduced to solve the problem that CNN cannot perform pixel-level fine segmentation(引入全卷积神经网络（FCN），解决CNN无法进行像素级精细分割的问题)
- [x] CGNet
- [ ] DeepLabV1
  Low-level vision task
+ Proposed Atrous Convolution , also named dilated convolution
+ Fully-connected Conditional Random Field , CRF

- [ ] DeepLabV2
+ Proposed Atrous Spatial Pyramid Pooling(ASPP) , which can be used to solve the problem of different detection target size differences by using dilated convolution with different dialation rate
+ Replace VGG used in DeepLabv1 with a deeper ResNet
- [ ] DeepLabV3
- [x] DeepLabV3+
+ Add decoder based on DeepLabv3 to restore object edge information
+ 
- [ ] HRNet 
### Detection
- [ ] Fast R-CNN
- [ ] Faster R-CNN
- [ ] YOLOv3
- [ ] YOLOv5
- [ ] YOLOX


### Classfication

## Support Loss
- [x] BCELoss
- [x] FocalLoss
- [x] DiceLoss

## Requirements
pytorch >= 1.9.0

## Training
```python
# train.sh
```

## Evaluation
```python
# val.sh
```
