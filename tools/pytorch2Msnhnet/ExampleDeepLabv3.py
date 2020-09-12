
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from PytorchToMsnhnet import *

deeplabv3=deeplabv3_resnet101(pretrained=False)
ccc = torch.load("C:/Users/msnh/.cache/torch/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth")
del ccc["aux_classifier.0.weight"]
del ccc["aux_classifier.1.weight"]
del ccc["aux_classifier.1.bias"]
del ccc["aux_classifier.1.running_mean"]
del ccc["aux_classifier.1.running_var"]
del ccc["aux_classifier.1.num_batches_tracked"]
del ccc["aux_classifier.4.weight"]
del ccc["aux_classifier.4.bias"]
deeplabv3.load_state_dict(ccc)
deeplabv3.requires_grad_(False)
deeplabv3.eval()


input=torch.ones([1,3,224,224])

''' 
# trans msnhnet file only  
transNet(deeplabv3, input, "deeplabv3.msnhnet")
'''

'''
# trans msnhbin file only  
transBin(deeplabv3, "deeplabv3.msnhbin")
'''

# trans msnhnet and msnhbin file
trans(deeplabv3, input,"deeplabv3.msnhnet","deeplabv3.msnhbin")