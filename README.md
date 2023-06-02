# OroModelZoo
[TOC]
## Introduction
  OroModelZoo is a PyTorch-based open source toolbox for deep learning network reproduction involving classification, segmentation, and detection of computer vision directions.  
  At present, it is mainly used for personal study and experiment of OroChi Fang.Welcome everyone to submit and provide constructive suggestions for network reproduction, thank you🤞

## Top News
**`2023-03`**:**Build Yolo**  
**`2023-02`**:**Build some classic networks and backbones**  
**`2023-01`**:**Learn about DistributeDataParallel training mode**
**`2022-12`**:**Split and Build training framework **
**`2022-10`**: **Create repository for OroModelZoo🎂**

## Supported model and backbone
### backbone
- [x] MobileNetv1
+ MobileNet is a lightweight deep nerural network proposed by Google for embedded devices such as mobile phones. 
+ Proposed Depthwise Separable Convolution:3 * 3 Depthwise Conv + 1 * 1 Pointwise Conv(深度可分离卷积).  
+ Use ReLU6 install of ReLU,The activation value can be distributed in a small range, and low-precision float16 and other embedded devices can be accurately described, thereby avoiding loss of precision.
+ Limitation:(1)No residual connections(2)Because the number of convolution kernel weights is too small,many Depthwise convolution kernels are trained to be 0.
- [x] MobileNetv2
+ Inverted Residual Block:Use 1x1 Poinwise Conv to increase the dimension before the 3x3 Depthwise Conv , and use 1x1 Poinwise Conv to reduce the dimension after the 3x3 Depthwise Conv, first expand and then compress(倒残差结构)
+ Linear Bottleneck: After using 1x1 convolution to reduce the dimensionality, the Relu6 layer is no longer performed, and the addition of the residual network is directly performed to achieve linear activation.(线性瓶颈)
- [x] MobileNetv3  
+ Modify the number of initial convolution kernels(224 * 224 * 3 -> 112 * 112 * 32 change to 224 * 224 * 3 -> 112 * 112 * 16)  
+ Squeeze-and-Excitation Network:Automatically obtain the importance of each feature through learning(SE-Net注意力机制)
+ Proposed H-Swish activate activation function  
+ Reconstruct the time-consuming structure 
+ Neural Architecture Search  
- [ ] ShuffleNetV1
+ Channel shuffle:,shuffle channel after this Conv(通道重排)
+ Pointwise Group Convolution:1 * 1Conv -> 1 * 1 Group Conv(逐点分组卷积)
- [ ] ShuffleNetV2
### Segmentation
- [x] UNet
+ One of the earlier algorithms for semantic segmentation tasks using multi-scale features
+ The fully convolutional neural network (FCN) is introduced to solve the problem that CNN cannot perform pixel-level fine segmentation(全卷积神经网络FCN)
- [x] CGNet
- [ ] DeepLabV1
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
- [x] YOLOv3
- [ ] YOLOv5
- [ ] YOLOX

## Backbone
- [ ] Vision Transformer

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

## License
This project is licensed under the MIT License.
