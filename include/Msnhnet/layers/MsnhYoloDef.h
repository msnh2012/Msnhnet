#ifndef MSNHYOLOINFO_H
#define MSNHYOLOINFO_H
#include <vector>
#include "Msnhnet/utils/MsnhTypes.h"
#include "Msnhnet/utils/MsnhException.h"

namespace Msnhnet
{
struct YoloInfo
{
    YoloInfo(const int& outHeight, const int& outWidth, const int& outChannel)
        :outHeight(outHeight),outWidth(outWidth),outChannel(outChannel){}

    int outHeight   =   0;
    int outWidth    =   0;
    int outChannel  =   0;

    int getOutputNum(){return outChannel*outWidth*outHeight;}
};

struct YoloBox
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
    YoloV3=30,
    YoloV3_ANGLE, 

    YoloV3_GAUSSIAN,
    YoloV4 = 40,
    YoloV4_ANGLE, 

    YoloV5 = 50,
    YoloV5_ANGLE, 

};

inline YoloType getYoloTypeFromStr(const std::string &str)
{
    if(str == "yolov3")
    {
        return YoloType::YoloV3;
    }
    else if(str == "yolov3Angle")
    {
        return YoloType::YoloV3_ANGLE;
    }
    else if(str == "yolov3Gaussian")
    {
        return YoloType::YoloV3_GAUSSIAN;
    }
    else if(str == "yolov4")
    {
        return YoloType::YoloV4;
    }
    else if(str == "yolov4Angle")
    {
        return YoloType::YoloV4_ANGLE;
    }
    else if(str == "yolov5")
    {
        return YoloType::YoloV5;
    }
    else if(str == "yolov5Angle")
    {
        return YoloType::YoloV5_ANGLE;
    }
    else
    {
        throw Msnhnet::Exception(1,str + " yolo type is not supported!",__FILE__,__LINE__,__FUNCTION__);
    }
}

inline std::string  getStrFromYoloType(const YoloType &type)
{
    switch (type) {
    case YoloV3:
        return "yolov3";
    case YoloV3_ANGLE:
        return "yolov3Angle";
    case YoloV3_GAUSSIAN:
        return "yolov3Gaussian";
    case YoloV4:
        return "yolov4";
    case YoloV4_ANGLE:
        return "yolov4Angle";
    case YoloV5:
        return "yolov5";
    case YoloV5_ANGLE:
        return "yolov5Angle";
    default:
        throw Msnhnet::Exception(1,std::to_string((int)type) + " yolo type is not supported!",__FILE__,__LINE__,__FUNCTION__);
    }
}

}

#endif 

