
import torch
import torch.nn as nn
from torchvision.models import resnet
from PytorchToMsnhnet import transNet
import numpy as np

# a = np.array((1,2,3,4,5,6,7,8,9,11,22,33,44,55,66,77,88,99,111,222,333,444,555,666,777,888,999))

# b = torch.from_numpy(a).reshape((1,3,3,3)).float()

# print(b)
# print(b.sum(2))

resnet18=resnet.resnet18(True)
resnet18.eval()
print(resnet18)
input=torch.ones([1,3,224,224])
transNet(resnet18, input, "resnet18.msnhnet", "resnet18.msnhbin")