from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from Keras2Msnh import *


model = ResNet50(weights='imagenet')
keras2Msnh(model,"resnet50.msnhnet", "resnet50.msnhbin")
