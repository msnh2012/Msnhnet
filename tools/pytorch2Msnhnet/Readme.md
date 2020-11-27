# Pytorch2msnhnet
## Attention!
Alpha version, maybe have some bugs. Only official op is supported, customized op may have some bugs.

## Supported OP
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

## API
- translate pytorch to msnhnet and msnhbin.
    ```def trans(net, inputVar, msnhnet_path, msnhbin_path)```
## Example:
```# Pytorch2msnhnet
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PytorchToMsnhnet import *

resnet18=resnet18(pretrained=True)
resnet18.eval()
input=torch.ones([1,3,224,224])
trans(resnet18, input,"resnet18.msnhnet","resnet18.msnhbin")
```
