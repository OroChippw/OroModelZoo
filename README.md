# OroModelZoo


## Supported model backbone
MobileNetv1  
  Depthwise Separable Convolution  
  局限：没有残差连接、很多Depthwise卷积核训练出来是0，因为卷积核权重数量太少，同时ReLU清零，低精度的浮点数  
  
MobileNetv2  
    Linear Bottleneck ; Inverted Residual  
  
MobileNetv3  
    更新了Block提出bneck  
        加入SE注意力机制，更新激活函数，  
    swish[x] 计算求导复杂不好量化
    h-sigmoid 
    h-swish[x]  
    Neural Architecture Search 搜索参数  
    重构耗时结构  
    Modify the number of channels of the head convolution kernel. Mobilenet v2 uses 32 x 3 x 3. The author found that 32 can actually be reduced a little bit, so here the author changed it to 16, which is reduced by 3ms on the premise of ensuring the accuracy. speed
    
ShuffleNetv1  
    Group Pointwise Convlution ：分组1*1卷积
    Limited : 近亲繁殖
    Channel Shuffle ：通道重排，实现跨组信息交流 
        Reshape成g行n列矩阵，转置，再进行Flatten
        可微分可导不引入额外参数
ShuffleNetv2  

