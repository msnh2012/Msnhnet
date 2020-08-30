#ifndef MSNHYOLOV3INFO_H
#define MSNHYOLOV3INFO_H
#include <vector>
#include "Msnhnet/utils/MsnhTypes.h"

namespace Msnhnet
{
struct Yolov3Info
{
    Yolov3Info(const int& outHeight, const int& outWidth, const int& outChannel)
        :outHeight(outHeight),outWidth(outWidth),outChannel(outChannel){}

    int outHeight   =   0;
    int outWidth    =   0;
    int outChannel  =   0;

    int getOutputNum(){return outChannel*outWidth*outHeight;}
};

struct Yolov3Box
{
    Box::XYWHBox xywhBox;
    float   conf            =   0;
    float   bestClsConf     =   0;
    int     bestClsIdx      =   0;
    float   angle           =   0;
    float   regAngle        =   0; 

    std::vector<float> classesVal;
    std::vector<float> angleSplits;
};

enum YoloType
{
    YoloV3_NORMAL,
    YoloV3_ANGLE,
    YoloV3_GAUSSIAN,
    YoloV4
};

}

#endif 

