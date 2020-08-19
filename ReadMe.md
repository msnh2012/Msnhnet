# Msnhnet
English | [中文](https://blog.csdn.net/MSNH2012/article/details/107216704)</br>
###  A mini pytorch inference framework which inspired from darknet.
![License](https://img.shields.io/badge/license-MIT-green)
![c++](https://img.shields.io/badge/lauguage-c%2B%2B-green)
![Msnhnet](https://img.shields.io/badge/Msnh-Msnhnet-blue)</br>
![](readme_imgs/banner.jpg)</br>
**OS supported** (you can check other OS by yourself)

| |windows|linux|mac|
|:---:|:---:|:---:|:---:|
|checked|![Windows](https://img.shields.io/badge/build-passing-brightgreen.svg)|![Windows](https://img.shields.io/badge/build-passing-brightgreen.svg)|![OSX](https://img.shields.io/badge/build-unknown-lightgrey.svg)|
|gpu|![Windows](https://img.shields.io/badge/build-passing-brightgreen.svg)|![Linux](https://img.shields.io/badge/build-passing-brightgreen.svg)|![Mac](https://img.shields.io/badge/build-unknown-lightgrey.svg)|

**CPU checked**
| |Intel i7|raspberry 3B|raspberry 4B|Jeston NX|
|:---:|:---:|:---:|:---:|:---:|
|checked|![i7](https://img.shields.io/badge/build-passing-brightgreen.svg)|![3B](https://img.shields.io/badge/build-passing-brightgreen.svg)|![4B](https://img.shields.io/badge/build-passing-brightgreen.svg)|![NX](https://img.shields.io/badge/build-passing-brightgreen.svg)|

**Features**

- C++ Only. 3rdparty blas lib is optional, also you can use OpenBlas.
- A viewer for msnhnet is supported.(netron like)
![](readme_imgs/msnhnetviewer.png)
- OS supported: Windows, Linux(Ubuntu checked) and Mac os(unchecked).
- CPU supported: Intel X86, AMD(unchecked) and ARM(checked: armv7 armv8 arrch64).
- Keras to Msnhnet is supported. (Keras 2 and tensorflow 1.x)
- GPU cuda supported.(Checked GTX1080Ti, Jetson NX)
- GPU cudnn supported.(Checked GTX1080Ti, Jetson NX)
- GPU fp16 mode supported.(Checked GTX1080Ti, Jetson NX.)
- **ps. Please check your card wheather fp16 full speed is supported.**
- Working on it...(**Weekend Only  (╮（╯＿╰）╭)**)

**Tested networks**
- lenet5
- lenet5_bn
- alexnet
- vgg16
- vgg16_bn
- resnet18
- resnet34
- resnet50
- resnet101
- resnet152
- darknet53
- googLenet
- mobilenetv2
- yolov3
- yolov3_spp
- yolov3_tiny
- yolov4
- fcns
- unet</br>
==============================================================
- mobilenetv2_yolov3_lite (cudnn does not work with GTX10** Pascal Card, please use GPU model only)
- mobilenetv2_yolov3_nano (cudnn does not work with GTX10** Pascal Card, please use GPU model only)
- yoloface100k (cudnn does not work with GTX10** Pascal Card, please use GPU model only)
- yoloface500k (cudnn does not work with GTX10** Pascal Card, please use GPU model only)
- Thanks: https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3
==============================================================
- **pretrained models** 链接：https://pan.baidu.com/s/1mBaJvGx7tp2ZsLKzT5ifOg 
提取码：x53z 

**Yolo Test** 
- Win10 MSVC 2017 I7-10700F

  |net|yolov3|yolov3_tiny|yolov4|
  |:---:|:---:|:---:|:---:|
  |time|465ms|75ms|600ms|


- ARM(Yolov3Tiny cpu)
  |cpu|raspberry 3B|raspberry 4B|Jeston NX|
  |:---:|:---:|:---:|:---:|
  |without NNPack|6s|2.5s|1.2s|
  |with NNPack|2.5s|1.1s|0.6s|

**Yolo GPU Test**
- Ubuntu16.04  GCC  Cuda10.1  GTX1080Ti
  |net|yolov3|yolov3_tiny|yolov4|
  |:---:|:---:|:---:|:---:|
  |time|30ms|8ms|30ms|
  
- Jetson NX
  |net|yolov3|yolov3_tiny|yolov4|
  |:---:|:---:|:---:|:---:|
  |time|280ms|30ms|210ms|
  
**Yolo GPU cuDnn FP16 Test**
- Jetson NX
  |net|yolov3|yolov4|
  |:---:|:---:|:---:|
  |time|140ms|120ms|
  
**Mobilenet Yolo GPU cuDnn Test**
- Jetson NX
  |net|yoloface100k|yoloface500k|mobilenetv2_yolov3_nano|mobilenetv2_yolov3_lite|
  |:---:|:---:|:---:|:---:|:---:|
  |time|7ms|20ms|20ms|30|
  
**Requirements**
  * OpenCV4 https://github.com/opencv/opencv
  * yaml-cpp https://github.com/jbeder/yaml-cpp
  * Qt5 (**optional**. for Msnhnet viewer) http://download.qt.io/archive/qt/
  * cuda10+ cudnn 7.0+.(for GPU)

**How to build**
- With CMake 3.10+
- Viewer can not build with GPU.
- Options</br>
![](readme_imgs/cmake_option.png)</br>
**ps. You can change omp threads by unchecking OMP_MAX_THREAD and modifying "num" val at CMakeLists.txt:43** </br>

- Windows
1. Compile opencv4 and yaml-cpp.
2. Config environment. Add "OpenCV_DIR" and "yaml-cpp_DIR" 
3. Get qt5 and install. http://download.qt.io/ **(optional)**
4. Add qt5 bin path to environment.
5. Then use cmake-gui tool and visual studio to make or use vcpkg.

- Linux(Ubuntu)

ps. If you want to build with Jetson, please uncheck NNPACK, OMP, OPENBLAS, NEON, OMP_MAX_THREAD.

```
sudo apt-get install qt5-default      #optional
sudo apt-get install libqt5svg5-dev   #optional
sudo apt-get install libopencv-dev

# build yaml-cpp
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mdir build 
cd build 
cmake ..
make -j4
sudo make install 

#config 
sudo echo /usr/local/lib > /etc/ld.so.conf/usrlib.conf
sudo ldconfig

# build Msnhnet
git clone https://github.com/msnh2012/Msnhnet.git

cd Msnhnet/build
cmake -DCMAKE_BUILD_TYPE=Release ..  
make -j4
sudo make install

vim ~/.bashrc # Last line add: export PATH=/usr/local/bin:$PATH

```
**Test Msnhnet**
- 1. Download pretrained model and extract. eg.D:/models. 
- 2. Open terminal and cd "Msnhnet install bin". eg. D:/Msnhnet/bin
- 3. Test yolov3 "yolov3 D:/models".
- 4. Test yolov3tiny_video "yolov3tiny_video D:/models".
- 5. Test classify "classify D:/models".</br>

![](readme_imgs/dog.png)</br>

**View Msnhnet**
- 1. Open terminal and cd "Msnhnet install bin" eg. D:/Msnhnet/bin
- 2. run "MsnhnetViewer"

![](readme_imgs/viewer.png)</br>

**PS. You can double click "ResBlock Res2Block AddBlock ConcatBlock"  node to view more detail**</br>
**ResBlock**</br>
![](readme_imgs/ResBlock.png)</br>

**Res2Block**</br>
![](readme_imgs/Res2Block.png)</br>

**AddBlock**</br>
![](readme_imgs/AddBlock.png)</br>

**ConcatBlock**</br>
![](readme_imgs/ConcatBlock.png)</br>

**How to convert your own pytorch network**
1. Use pytorch to load network
```
import torchvision.models as models
import torch
from torchsummary import summary 

md = models.resnet18(pretrained = True)
md.to("cpu")
md.eval()

print(md, file = open("net.txt", "a"))

summary(md, (3, 224, 224),device='cpu')
```
2. Write msnhnet file according to net.txt and summary result.(Manually :o. Like darnet cfg)
3. Export msnhbin 
```
val = []
dd = 0
for name in md.state_dict():
        if "num_batches_tracked" not in name:
                c = md.state_dict()[name].data.flatten().numpy().tolist()
                dd = dd + len(c)
                print(name, ":", len(c))
                val.extend(c)

with open("alexnet.msnhbin","wb") as f:
    for i in val :
        f.write(pack('f',i))
```
**Ps. More detail in file "pytorch2msnhbin/pytorch2msnhbin.py"**

**About Train**
- Just use pytorch to train your model, and export as msnhbin.
- eg. yolov3/v4 [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)

Enjoy it! :D

**加群交流**</br>
![](readme_imgs/qq.png)</br>
