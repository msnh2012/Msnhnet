## Keras to Msnhnet
---
**Requirements**
- Keras 2 and tensorflow 1.x
  
**How to use.**
```
keras2Msnh(model,"resnet50.msnhnet", "resnet50.msnhbin")
```
**Supported Layers**

- InputLayer
- Conv2D/Convolution2D
- DepthwiseConv2D
- MaxPooling2D
- AveragePooling2D
- BatchNormalization
- LeakyReLU
- Activation(relu, relu6, leakyReLU, sigmoid, linear)
- UpSampling2D
- Concatenate/Merge
- Add
- ZeroPadding2D
- GlobalAveragePooling2D
- softmax
- Dense