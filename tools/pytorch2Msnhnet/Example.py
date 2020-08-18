
import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.segmentation import deeplabv3_resnet101
from PytorchToMsnhnet import transNet


deeplab = deeplabv3_resnet101()
print(deeplab,file = open("net.txt", "a"))

resnet18=resnet.resnet18(True)
resnet18.eval()
print(resnet18)
input=torch.ones([1,3,224,224])

# for name, module in resnet18.named_modules():
#     print('modules:', name)


transNet(resnet18, input, "resnet18.msnhnet", "resnet18.msnhbin")
