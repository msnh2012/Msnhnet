import os
import io

import onnx
import numpy as np
import torch
import onnxruntime as ort
import argparse

from onnx2pytorch import convert
from PytorchToMsnhnet import *

def convert_onnx_msnhnet(onnx_model, pytorch_model, onnx_model_outputs, onnx_inputs):
    model = convert.ConvertModel(onnx_model)
    model.eval()
    model.cpu()
    
    with torch.no_grad():
        outputs = model(onnx_inputs)

    
parser = argparse.ArgumentParser(description='Convert onnx model to MsnhNet model.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--height', type=int, default=None)
parser.add_argument('--width', type=int, default=None)
parser.add_argument('--channels', type=int, default=None)

args = parser.parse_args()

model_def = args.model
name = model_def.split('/')[-1].split('.')[0]
width = args.width
height = args.height
channels = args.channels

onnx_model = onnx.load(model_def)

model = convert.ConvertModel(onnx_model)
model.eval()
model.cpu()

input=torch.ones([1,channels,height,width])

model_name = name + ".msnhnet"

model_bin = name + ".msnhbin"

trans(model, input,model_name,model_bin)
