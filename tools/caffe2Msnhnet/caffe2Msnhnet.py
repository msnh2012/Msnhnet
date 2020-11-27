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


