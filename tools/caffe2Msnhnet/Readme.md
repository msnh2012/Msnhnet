# Caffe2msnhnet

# 介绍

Caffe2msnhnet工具首先将你的Caffe模型转换为Pytorch模型，然后调用Pytorch2msnhnet工具将Caffe模型转为`*.msnhnet`和`*.bin`。

## 依赖
- Pycaffe
- Pytorch 



# 计算图优化

- 在调用`caffe2msnhnet.py`之前建议使用caffeOPtimize文件夹中的`caffeOptimize.py`对原始的Caffe模型进行图优化，目前已支持的操作有：

- [x] Conv+BN+Scale 融合到 Conv
- [x] Deconv+BN+Scale 融合到Deconv
- [x] InnerProduct+BN+Scale 融合到InnerProduct

## Caffe2Pytorch支持的OP
- [x] Convolution 转为 `nn.Conv2d`
- [x] Deconvolution 转为 `nn.ConvTranspose2d`
- [x] BatchNorm 转为 `nn.BatchNorm2d或者nn.BatchNorm1d`
- [x] Scale 转为 `乘/加`
- [x] ReLU 转为 `nn.ReLU`
- [x] LeakyReLU 转为 `nn.LeakyReLU`
- [x] PReLU 转为 `nn.PReLU`
- [x] Max Pooling 转为 `nn.MaxPool2d`
- [x] AVE Pooling 转为 `nn.AvgPool2d`
- [x] Eltwise 转为 `加/减/乘/除/torch.max`
- [x] InnerProduct 转为`nn.Linear`
- [x] Normalize 转为 `pow/sum/sqrt/加/乘/除`拼接
- [x] Permute 转为`torch.permute`
- [x] Flatten 转为`torch.view`
- [x] Reshape 转为`numpy.reshape/torch.from_numpy`拼接
- [x] Slice 转为`torch.index_select`
- [x] Concat 转为`torch.cat`
- [x] Crop 转为`torch.arange/torch.resize_`拼接
- [x] Softmax 转为`torch.nn.function.softmax`



# Pytorch2Msnhnet支持的OP

- [x] conv2d
- [x] max_pool2d
- [x] avg_pool2d
- [x] adaptive_avg_pool2d
- [x] linear
- [x] flatten
- [x] dropout
- [x] batch_norm
- [x] interpolate(nearest, bilinear)
- [x] cat   
- [x] elu
- [x] selu
- [x] relu
- [x] relu6
- [x] leaky_relu
- [x] tanh
- [x] softmax
- [x] sigmoid
- [x] softplus
- [x] abs    
- [x] acos   
- [x] asin   
- [x] atan   
- [x] cos    
- [x] cosh   
- [x] sin    
- [x] sinh   
- [x] tan    
- [x] exp    
- [x] log    
- [x] log10  
- [x] mean
- [x] permute
- [x] view
- [x] contiguous
- [x] sqrt
- [x] pow
- [x] sum
- [x] pad
- [x] +|-|x|/|+=|-=|x=|/=|



## 使用方法举例
- `python caffe2msnhnet  --model  landmark106.prototxt --weights landmark106.caffemodel --height 112 --width 112 --channels 3 `，执行完之后会在当前目录下生成`lanmark106.msnhnet`和`landmark106.bin`文件。



## caffe2msnhnet示例

```c++
# -*- coding: utf-8
# from pytorch2caffe import plot_graph, pytorch2caffe
import sys
import cv2
import caffe
import numpy as np
import os
from caffenet import *
import argparse
import torch
from PytorchToMsnhnet import *

################################################################################################   
parser = argparse.ArgumentParser(description='Convert Caffe model to MsnhNet model.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--height', type=int, default=None)
parser.add_argument('--width', type=int, default=None)
parser.add_argument('--channels', type=int, default=None)

args = parser.parse_args()

model_def = args.model
model_weights = args.weights
name = model_weights.split('/')[-1].split('.')[0]
width = args.width
height = args.height
channels = args.channels


net = CaffeNet(model_def, width=width, height=height, channels=channels)
net.load_weights(model_weights)
net.to('cpu')
net.eval()

input=torch.ones([1,channels,height,width])

model_name = name + ".msnhnet"

model_bin = name + ".msnhbin"

trans(net, input,model_name,model_bin)
```



# 参考

- https://github.com/UltronAI/pytorch-caffe