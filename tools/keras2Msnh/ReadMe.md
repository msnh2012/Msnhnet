## Keras to Msnhnet
---
**Requirements**
- Keras 2 and tensorflow 1.x
  
**How to use.**
```
keras2Msnh(model,"resnet50.msnhnet", "resnet50.msnhbin")
```
**Supported Layers**

- [x] InputLayer
- [x] Conv2D/Convolution2D
- [x] DepthwiseConv2D
- [x] MaxPooling2D
- [x] AveragePooling2D
- [x] BatchNormalization
- [x] LeakyReLU
- [x] Activation(relu, relu6, leakyReLU, sigmoid, linear)
- [x] UpSampling2D
- [x] Concatenate/Merge
- [x] Add
- [x] ZeroPadding2D
- [x] GlobalAveragePooling2D
- [x] softmax
- [x] Dense