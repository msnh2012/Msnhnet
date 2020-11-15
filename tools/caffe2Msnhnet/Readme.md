# Caffe2msnhnet

# 介绍

Caffe2msnhnet工具首先将你的Caffe模型转换为Pytorch模型，然后调用Pytorch2msnhnet工具将Caffe模型转为`*.msnhnet`和`*.bin`。

## 依赖
- Pycaffe
- Pytorch 

## Caffe2Pytorch支持的OP
- Convolution 转为 `nn.Conv2d`
- Deconvolution 转为 `nn.ConvTranspose2d`
- BatchNorm 转为 `nn.BatchNorm2d或者nn.BatchNorm1d`
- Scale 转为 `乘/加`
- ReLU 转为 `nn.ReLU`
- LeakyReLU 转为 `nn.LeakyReLU`
- PReLU 转为 `nn.PReLU`
- Max Pooling 转为 `nn.MaxPool2d`
- AVE Pooling 转为 `nn.AvgPool2d`
- Eltwise 转为 `加/减/乘/除/torch.max`
- InnerProduct 转为`nn.Linear`
- Normalize 转为 `pow/sum/sqrt/加/乘/除`拼接
- Permute 转为`torch.permute`
- Flatten 转为`torch.view`
- Reshape 转为`numpy.reshape/torch.from_numpy`拼接
- Slice 转为`torch.index_select`
- Concat 转为`torch.cat`
- Crop 转为`torch.arange/torch.resize_`拼接
- Softmax 转为`torch.nn.function.softmax`



# Pytorch2MsnhNet支持的OP

- conv2d
- max_pool2d
- avg_pool2d
- adaptive_avg_pool2d
- linear
- flatten
- dropout
- batch_norm
- interpolate(nearest, bilinear)
- cat   
- elu
- selu
- relu
- relu6
- leaky_relu
- tanh
- softmax
- sigmoid
- softplus
- abs    
- acos   
- asin   
- atan   
- cos    
- cosh   
- sin    
- sinh   
- tan    
- exp    
- log    
- log10  
- mean
- permute
- view
- contiguous
- sqrt
- pow
- sum
- pad
- +|-|x|/|+=|-=|x=|/=|



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