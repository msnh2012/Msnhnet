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

struct Yolov3FinalBox
{
    Yolov3FinalBox(const float &x, const float &y, const float &w, const float &h, const float &conf, const int &bestCls)
                :x(x),y(y),w(w),h(h),conf(conf),bestCls(bestCls){}
    float    x      =   0;
    float    y      =   0;
    float    w      =   0;
    float    h      =   0;
    float    conf   =   0;
    int      bestCls=   0;

};

struct Yolov3Box
{
    Box::XYWHBox xywhBox;
    float   conf            =   0;
    float   bestClsConf     =   0;
    int     bestClsIdx  =   0;
    std::vector<float> classesVal;
};

}

#endif 

