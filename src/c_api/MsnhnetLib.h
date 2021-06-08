#ifndef MSNHNETLIB_H
#define MSNHNETLIB_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/Msnhnet.h"
#include <iostream>
#include <memory>

const int MaxBBoxNum = 1024;

enum PredDataType
{
    PRE_DATA_NONE = 0,
    PRE_DATA_FC32_C1,
    PRE_DATA_FC32_C3,
    PRE_DATA_GOOGLENET_FC3,
    PRE_DATA_PADDING_ZERO_FC3,
    PRE_DATA_TRANSFORMED_FC3,
    PRE_DATA_CAFFE_FC3
};

struct BBox
{
    float       x            =   0;
    float       y            =   0;
    float       w            =   0;
    float       h            =   0;
    float       conf         =   0;
    float       bestClsConf  =   0;
    uint32_t    bestClsIdx   =   0;
    float       angle        =   0;
};

struct BBoxContainer
{
    BBox boxes[MaxBBoxNum];
};

extern "C" MsnhNet_API int initMsnhnet();
extern "C" MsnhNet_API int buildMsnhnet(char **msg, const char* msnhnet, const char* msnhbin, int useFp16=0, int useCudaOnly=0);
extern "C" MsnhNet_API int runClassifyFile(char **msg, const char* imagePath, int* bestIndex,  PredDataType preDataType=PRE_DATA_NONE,
                                           int runGPU = 0, const float* mean=nullptr, const float *std=nullptr);

extern "C" MsnhNet_API int runClassifyList(char **msg, unsigned char* imageData, const int width, const int height, const int channel,
                                           int* bestIndex,  PredDataType preDataType=PRE_DATA_NONE, const int swapRGB=false, int runGPU = 0,
                                           const float* mean=nullptr, const float *std=nullptr);

extern "C" MsnhNet_API int runClassifyNoPred(char** msg, const float *data, const int len, int *bestIndex, int runGPU = 0);

extern "C" MsnhNet_API int runYoloFile(char **msg, const char* imagePath, BBoxContainer *bboxContainer, int *detectedNum, const int runGPU = 0);

extern "C" MsnhNet_API int runYoloList(char **msg, unsigned char* imageData, const int width, const int height, const int channel, BBoxContainer *bboxContainer,
                                       int *detectedNum, const int swapRGB, const int runGPU = 0);

extern "C" MsnhNet_API int dispose();
extern "C" MsnhNet_API int withGPU(int *GPU);
extern "C" MsnhNet_API int withCUDNN(int *CUDNN);

extern "C" MsnhNet_API int getCpuForwardTime(float *time);
extern "C" MsnhNet_API int getGpuForwardTime(float *time);
extern "C" MsnhNet_API int getInputDim(int *width, int *heigth, int *channel);
#endif
