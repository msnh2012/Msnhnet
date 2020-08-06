#ifndef MSNHPASER_H
#define MSNHPASER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include <yaml-cpp/yaml.h>
#include "Msnhnet/utils/MsnhExString.h"
#include "Msnhnet/layers/MsnhActivations.h"
#include "Msnhnet/layers/MsnhYolov3Def.h"
#include "Msnhnet/utils/MsnhTypes.h"
#include <string>
#include <fstream>
#include "Msnhnet/utils/MsnhExport.h"

namespace  Msnhnet
{
class MsnhNet_API BaseParams
{
public:
    BaseParams(bool incIndex)
    {
        if(incIndex)
        {
            index++;
        }
    }
    LayerType type;
    static int index;
};

class NetConfigParams : public BaseParams
{
public:
    NetConfigParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type= LayerType::CONFIG;
    }
    int             batch       =   1;
    int             width       =   0;
    int             height      =   0;
    int             channels    =   0;
};

class EmptyParams : public BaseParams
{
public:
    EmptyParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::EMPTY;
    }
};

class ActivationParams : public BaseParams
{
public:
    ActivationParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::ACTIVE;
    }
    ActivationType activation = ActivationType::NONE;
};

class ConvParams : public  BaseParams
{
public:
    ConvParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::CONVOLUTIONAL;
    }
    int             batchNorm   =   0;
    int             filters     =   1;
    int             groups      =   1;

    int             kSize       =   1;
    int             kSizeX      =   -1;
    int             kSizeY      =   -1;

    int             stride      =   1;
    int             strideX     =   -1;
    int             strideY     =   -1;
    int             antialiasing=   0;

    int             padding     =   0;
    int             paddingX    =   -1;
    int             paddingY    =   -1;

    int             dilation    =   1;
    int             dilationX   =   -1;
    int             dilationY   =   -1;
    int             useBias     =   1; 

    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;
};

class DeConvParams : public  BaseParams
{
public:
    DeConvParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::DECONVOLUTIONAL;
    }

    int             filters     =   1;

    int             kSize       =   1;
    int             kSizeX      =   -1;
    int             kSizeY      =   -1;

    int             stride      =   1;
    int             strideX     =   -1;
    int             strideY     =   -1;

    int             padding     =   0;
    int             paddingX    =   -1;
    int             paddingY    =   -1;

    int             groups      =   1;

    int             useBias     =   1; 

    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;
};

class MaxPoolParams : public BaseParams
{
public:
    MaxPoolParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type  = LayerType::MAXPOOL;
    }
    int             kSize       =   1;
    int             kSizeX      =   -1;
    int             kSizeY      =   -1;

    int             stride      =   1;
    int             strideX     =   -1;
    int             strideY     =   -1;

    int             padding     =   0;
    int             paddingX    =   -1;
    int             paddingY    =   -1;

    int             maxPoolDepth=   0;
    int             outChannels =   1;

    int             ceilMode   =   0;
    int             global      =   0;
};

class LocalAvgPoolParams : public BaseParams
{
public:
    LocalAvgPoolParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type  = LayerType::LOCAL_AVGPOOL;
    }
    int             kSize       =   1;
    int             kSizeX      =   -1;
    int             kSizeY      =   -1;

    int             stride      =   1;
    int             strideX     =   -1;
    int             strideY     =   -1;

    int             padding     =   0;
    int             paddingX    =   -1;
    int             paddingY    =   -1;

    int             ceilMode    =   0;
    int             global      =   0;
};

class GlobalAvgPoolParams : public BaseParams
{
public:
    GlobalAvgPoolParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type  = LayerType::GLOBAL_AVGPOOL;
    }
};

class ConnectParams : public BaseParams
{
public:
    ConnectParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type  = LayerType::CONNECTED;
    }
    int             output      =   0;
    int             batchNorm   =   0;
    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;
};

class BatchNormParams : public BaseParams
{
public:
    BatchNormParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type= LayerType::BATCHNORM;
    }
    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;
};

class MsnhNet_API ResBlockParams : public BaseParams
{
public:
    ResBlockParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::RES_BLOCK;
    }

    std::vector<BaseParams* > baseParams;
    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;
    ~ResBlockParams();

};

class MsnhNet_API Res2BlockParams : public BaseParams
{
public:
    Res2BlockParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type= LayerType::RES_2_BLOCK;
    }

    std::vector<BaseParams* > baseParams;
    std::vector<BaseParams* > branchParams;
    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;

    ~Res2BlockParams();
};

class MsnhNet_API ConcatBlockParams : public BaseParams
{
public:
    ConcatBlockParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type= LayerType::CONCAT_BLOCK;
    }
    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;

    std::vector<std::vector<BaseParams* >> branchParams;

    ~ConcatBlockParams();
};

class MsnhNet_API AddBlockParams : public BaseParams
{
public:
    AddBlockParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type= LayerType::ADD_BLOCK;
    }
    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;

    std::vector<std::vector<BaseParams* >> branchParams;

    ~AddBlockParams();
};

class RouteParams : public BaseParams
{
public:
    RouteParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::ROUTE;
    }
    std::vector<int> layerIndexes;

    int     groups      =   1;
    int     groupsId    =   0;
    int     addModel    =   0;
};

class SoftMaxParams : public BaseParams
{
public:
    SoftMaxParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::SOFTMAX;
    }

    int     groups      =   1;
    float   temperature =   1;
};

class UpSampleParams : public BaseParams
{
public:
    UpSampleParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::UPSAMPLE;
    }
    int     stride      =   2;
    float   scale       =   1.f;
};

class Yolov3Params : public BaseParams
{
public:
    Yolov3Params(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::YOLOV3;
    }

    int     orgWidth    =   0;
    int     orgHeight   =   0;
    int     classNum    =   1;
    std::vector<float> anchors;
};

class Yolov3OutParams : public BaseParams
{
public:
    Yolov3OutParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::YOLOV3_OUT;
    }
    YoloType  yoloType  =   YoloType::YoloV3_NORMAL;
    int     orgWidth    =   0;
    int     orgHeight   =   0;
    float   confThresh  =   0;
    float   nmsThresh   =   0;
    int     useSoftNms  =   0;
    std::vector<int> layerIndexes;
    void getYoloTypeFromStr(const std::string &str)
    {
        if(str == "yolov3Normal")
        {
            yoloType = YoloType::YoloV3_NORMAL;
        }
        else if(str == "yolov3Angle")
        {
            yoloType = YoloType::YoloV3_ANGLE;
        }
        else if(str == "yolov3Gaussian")
        {
            yoloType = YoloType::YoloV3_GAUSSIAN;
        }
        else if(str == "yolov4")
        {
            yoloType = YoloType::YoloV4;
        }
    }
};

class PaddingParams : public BaseParams
{
public:
    PaddingParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::PADDING;
    }

    int     top         =   0;
    int     down        =   0;
    int     left        =   0;
    int     right       =   0;
    float   paddingVal  =   0;
};

class MsnhNet_API Parser
{
public:
    Parser();
    ~Parser();

    std::vector<BaseParams* >   params;
    std::vector<float>          msnhF32Weights;

    void clearParams();
    void readCfg(const std::string &path);
    void readMsnhBin(const std::string &path);

    void parseConfigParams(NetConfigParams *netConfigParams, YAML::const_iterator &iter);
    void parseActivationParams(ActivationParams *activationParams, YAML::const_iterator &iter);
    void parseMaxPoolParams(MaxPoolParams *maxPoolParams, YAML::const_iterator &iter);
    void parseLocalAvgPoolParams(LocalAvgPoolParams *localAvgPoolParams, YAML::const_iterator &iter);
    void parseGlobalAvgPoolParams(GlobalAvgPoolParams *globalAvgPoolParams, YAML::const_iterator &iter);
    void parseConvParams(ConvParams *convParams, YAML::const_iterator &iter);
    void parseDeConvParams(DeConvParams *deconvParams, YAML::const_iterator &iter);
    void parseConnectParams(ConnectParams *connectParams, YAML::const_iterator &iter);
    void parseBatchNormParams(BatchNormParams *batchNormParams, YAML::const_iterator &iter);
    void parseEmptyNormParams(EmptyParams *emptyParams, YAML::const_iterator &iter);
    void parsePaddingParams(PaddingParams *paddingParams, YAML::const_iterator &iter);
    void parseResBlockParams(ResBlockParams *resBlockParams, YAML::const_iterator &iter);
    void parseRes2BlockParams(Res2BlockParams *res2BlockParams, YAML::const_iterator &iter);
    void parseConcatBlockParams(ConcatBlockParams *concatBlockParams, YAML::const_iterator &iter);
    void parseAddBlockParams(AddBlockParams *addBlockParams, YAML::const_iterator &iter);
    void parseRouteParams(RouteParams *routeParams, YAML::const_iterator &iter);
    void parseSoftMaxParams(SoftMaxParams *softmaxParams, YAML::const_iterator &iter);
    void parseUpSampleParams(UpSampleParams *upSampleParams, YAML::const_iterator &iter);
    void parseYolov3Params(Yolov3Params *yolov3Params, YAML::const_iterator &iter);
    void parseYolov3OutParams(Yolov3OutParams *yolov3OutParams, YAML::const_iterator &iter);

};
}

#endif 

