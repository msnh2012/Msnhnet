
import torch
import torch.nn as nn
from torchvision.models import resnet
from PytorchToMsnhnet import transNet

resnet18=resnet.resnet18(True)
resnet18.eval()
print(resnet18)
input=torch.ones([1,3,224,224])
transNet(resnet18, input, "resnet18.msnhnet", "resnet18.msnhbin")