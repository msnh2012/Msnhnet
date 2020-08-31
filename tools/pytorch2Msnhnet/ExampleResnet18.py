
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PytorchToMsnhnet import *

resnet18=resnet18(pretrained=True)
resnet18.eval()
input=torch.ones([1,3,224,224])

''' 
# trans msnhnet file only  
transNet(resnet18, input, "resnet18.msnhnet")
'''

'''
# trans msnhbin file only  
transBin(resnet18, "resnet18.msnhbin")
'''

# trans msnhnet and msnhbin file
trans(resnet18, input,"resnet18.msnhnet","resnet18.msnhbin")
