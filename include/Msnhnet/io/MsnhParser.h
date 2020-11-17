#ifndef MSNHPASER_H
#define MSNHPASER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/3rdparty/yaml-cpp/yaml.h"
#include "Msnhnet/utils/MsnhExString.h"
#include "Msnhnet/layers/MsnhActivations.h"
#include "Msnhnet/layers/MsnhYoloDef.h"
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

class MsnhNet_API NetConfigParams : public BaseParams
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

class MsnhNet_API EmptyParams : public BaseParams
{
public:
    EmptyParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::EMPTY;
    }
};

class MsnhNet_API ViewParams : public BaseParams
{
public:
    ViewParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::VIEW;
    }
    int             dim0        =   -1;
    int             dim1        =   -1;
    int             dim2        =   -1;
};

class MsnhNet_API PermuteParams : public BaseParams
{
public:
    PermuteParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::PERMUTE;
    }
    int             dim0        =   0;
    int             dim1        =   1;
    int             dim2        =   2;
};

class MsnhNet_API PixshuffleParams : public BaseParams
{
public:
    PixshuffleParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::PIXEL_SHUFFLE;
    }
    int             factor      =   0;
};

class MsnhNet_API SliceParams : public BaseParams
{
public:
    SliceParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::SLICE;
    }

    int             start0      =   0;
    int             step0       =   1;
    int             start1      =   0;
    int             step1       =   1;
    int             start2      =   0;
    int             step2       =   1;
};

class MsnhNet_API ReductionParams : public BaseParams
{
public:

    ReductionParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::REDUCTION;
    }
    int             axis        =   -1;
    ReductionType   reduceType  =   ReductionType::REDUCTION_SUM;

    static std::string getStrFromReduceType(ReductionType type);
    static ReductionType getReduceTypeFromStr(std::string typeStr);
};

class MsnhNet_API ActivationParams : public BaseParams
{
public:
    ActivationParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type     = LayerType::ACTIVE;
    }
    ActivationType activation = ActivationType::NONE;
    std::vector<float> actParams;
};

class MsnhNet_API ConvParams : public  BaseParams
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
    float           bnEps       =   0.00001f;
};

class MsnhNet_API DeConvParams : public  BaseParams
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

class MsnhNet_API MaxPoolParams : public BaseParams
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

class MsnhNet_API LocalAvgPoolParams : public BaseParams
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

class MsnhNet_API GlobalAvgPoolParams : public BaseParams
{
public:
    GlobalAvgPoolParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type  = LayerType::GLOBAL_AVGPOOL;
    }
};

class MsnhNet_API ConnectParams : public BaseParams
{
public:
    ConnectParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type  = LayerType::CONNECTED;
    }
    int             output      =   0;
    int             batchNorm   =   0;
    int             useBias     =   1; 

    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;
    float           bnEps       =   0.00001f;
};

class MsnhNet_API BatchNormParams : public BaseParams
{
public:
    BatchNormParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type= LayerType::BATCHNORM;
    }
    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;
    float eps   =   0.00001f;
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

class MsnhNet_API RouteParams : public BaseParams
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
    ActivationType  activation  =   ActivationType::NONE;
    std::vector<float> actParams;
};

class MsnhNet_API VariableOpParams : public BaseParams
{
public:
    VariableOpParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::VARIABLE_OP;
    }

    enum VarOpType
    {

        VAR_OP_ADD=0,

        VAR_OP_SUB,

        VAR_OP_SUB_INV,

        VAR_OP_MUL,

        VAR_OP_DIV,

        VAR_OP_DIV_INV,

        VAR_OP_ADD_CONST,

        VAR_OP_SUB_CONST,

        VAR_OP_SUB_CONST_INV,

        VAR_OP_MUL_CONST,

        VAR_OP_DIV_CONST,

        VAR_OP_DIV_CONST_INV,

        VAR_OP_ABS,

        VAR_OP_ACOS,

        VAR_OP_ASIN,

        VAR_OP_ATAN,

        VAR_OP_COS,

        VAR_OP_COSH,

        VAR_OP_SIN,

        VAR_OP_SINH,

        VAR_OP_TAN,

        VAR_OP_TANH,

        VAR_OP_EXP,

        VAR_OP_POW,

        VAR_OP_LOG,

        VAR_OP_LOG10,

        VAR_OP_SQRT,

    };

    std::vector<int> layerIndexes;
    VarOpType varOpType     =   VAR_OP_ADD;
    float constVal          =   0;

    static std::string getStrFromVarOpType(const VarOpType &varOpType);
    static VarOpType   getVarOpTypeFromStr(const std::string &varOpStr);
};

class MsnhNet_API SoftMaxParams : public BaseParams
{
public:
    SoftMaxParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::SOFTMAX;
    }

    int     groups      =   1;
    float   temperature =   1;
};

class MsnhNet_API UpSampleParams : public BaseParams
{
public:
    UpSampleParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::UPSAMPLE;
    }
    enum UpsampleType
    {
        NEAREST = 0,
        BILINEAR,
    };

    int     strideX      =   -1;
    int     strideY      =   -1;
    int     stride       =   1;
    int     alignCorners =   0;
    float   scale        =   1.f;
    float   scaleX       =   -1.f;
    float   scaleY       =   -1.f;

    UpsampleType upsampleType = NEAREST;
    static UpsampleType getUnsampleTypeFromStr(const std::string &str);
    static std::string  getStrFromUnsampleType(const UpsampleType &type);
};

class MsnhNet_API YoloParams : public BaseParams
{
public:
    YoloParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::YOLO;
    }

    int     orgWidth    =   0;
    int     orgHeight   =   0;
    int     classNum    =   1;
    std::vector<float> anchors;
    YoloType yoloType   =   YoloType::YoloV3;
};

class MsnhNet_API YoloOutParams : public BaseParams
{
public:
    YoloOutParams(bool incIndex) : BaseParams(incIndex)
    {
        this->type = LayerType::YOLO_OUT;
    }
    int     orgWidth    =   0;
    int     orgHeight   =   0;
    float   confThresh  =   0;
    float   nmsThresh   =   0;
    int     useSoftNms  =   0;
    std::vector<int> layerIndexes;
    YoloType  yoloType  =   YoloType::YoloV3;
};

class MsnhNet_API PaddingParams : public BaseParams
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
    void parseEmptyParams(EmptyParams *emptyParams, YAML::const_iterator &iter);
    void parseViewParams(ViewParams *viewParams, YAML::const_iterator &iter);
    void parsePermuteParams(PermuteParams *permuteParams, YAML::const_iterator &iter);
    void parsePixShuffleParams(PixshuffleParams *pixShuffleParams, YAML::const_iterator &iter);
    void parseSliceParams(SliceParams *sliceParams, YAML::const_iterator &iter);
    void parsePaddingParams(PaddingParams *paddingParams, YAML::const_iterator &iter);
    void parseReductionParams(ReductionParams *reductionParams, YAML::const_iterator &iter);
    void parseResBlockParams(ResBlockParams *resBlockParams, YAML::const_iterator &iter);
    void parseRes2BlockParams(Res2BlockParams *res2BlockParams, YAML::const_iterator &iter);
    void parseConcatBlockParams(ConcatBlockParams *concatBlockParams, YAML::const_iterator &iter);
    void parseAddBlockParams(AddBlockParams *addBlockParams, YAML::const_iterator &iter);
    void parseRouteParams(RouteParams *routeParams, YAML::const_iterator &iter);
    void parseVariableOpParams(VariableOpParams *variableOpParams, YAML::const_iterator &iter);
    void parseSoftMaxParams(SoftMaxParams *softmaxParams, YAML::const_iterator &iter);
    void parseUpSampleParams(UpSampleParams *upSampleParams, YAML::const_iterator &iter);
    void parseYoloParams(YoloParams *yoloParams, YAML::const_iterator &iter);
    void parseYoloOutParams(YoloOutParams *yoloOutParams, YAML::const_iterator &iter);

};
}

#endif 

