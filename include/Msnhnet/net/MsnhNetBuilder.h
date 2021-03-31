#ifndef MSNHNETBUILDER_H
#define MSNHNETBUILDER_H

#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/io/MsnhParser.h"
#include "Msnhnet/layers/MsnhActivationLayer.h"
#include "Msnhnet/layers/MsnhBatchNormLayer.h"
#include "Msnhnet/layers/MsnhConnectedLayer.h"
#include "Msnhnet/layers/MsnhConvolutionalLayer.h"
#include "Msnhnet/layers/MsnhCropLayer.h"
#include "Msnhnet/layers/MsnhDeConvolutionalLayer.h"
#include "Msnhnet/layers/MsnhLocalAvgPoolLayer.h"
#include "Msnhnet/layers/MsnhGlobalAvgPoolLayer.h"
#include "Msnhnet/layers/MsnhMaxPoolLayer.h"
#include "Msnhnet/layers/MsnhRouteLayer.h"
#include "Msnhnet/layers/MsnhVariableOpLayer.h"
#include "Msnhnet/layers/MsnhEmptyLayer.h"
#include "Msnhnet/layers/MsnhViewLayer.h"
#include "Msnhnet/layers/MsnhPermuteLayer.h"
#include "Msnhnet/layers/MsnhSliceLayer.h"
#include "Msnhnet/layers/MsnhReductionLayer.h"
#include "Msnhnet/layers/MsnhSoftMaxLayer.h"
#include "Msnhnet/layers/MsnhUpSampleLayer.h"
#include "Msnhnet/layers/MsnhResBlockLayer.h"
#include "Msnhnet/layers/MsnhRes2BlockLayer.h"
#include "Msnhnet/layers/MsnhAddBlockLayer.h"
#include "Msnhnet/layers/MsnhConcatBlockLayer.h"
#include "Msnhnet/layers/MsnhYoloLayer.h"
#include "Msnhnet/layers/MsnhYoloOutLayer.h"
#include "Msnhnet/layers/MsnhPaddingLayer.h"
#include "Msnhnet/layers/MsnhPixelShuffLeLayer.h"
#include "Msnhnet/io/MsnhIO.h"
#include "Msnhnet/utils/MsnhExport.h"

#ifdef USE_OPENCL
#include "Msnhnet/core/cl/clScheduler.h"
#endif

namespace Msnhnet
{
class NetBuildParams
{
public:
    int batch       =   0;
    int inputNums   =   0;
    int height      =   0;
    int width       =   0;
    int channels    =   0;
};

class MsnhNet_API NetBuilder
{
public:
    NetBuilder();
    ~NetBuilder();

    static void setPreviewMode(const bool &mode);

    static void setMemAlign(const bool &memAlign);

#ifdef USE_GPU
    static void setOnlyGpu(const bool &onlyGpu);
    static void setOnlyCpu(const bool &onlyCpu);
    static void setUseFp16(const bool &useFp16);
    static void setForceUseCuda(const bool &onlyUseCuda);
#endif

    void buildNetFromMsnhNet(const std::string &path);
    void loadWeightsFromMsnhBin(const std::string &path);

    std::vector<float> runClassify(std::vector<float> img);
    std::vector<std::vector<YoloBox>> runYolo(std::vector<float> img);
#ifdef USE_GPU
    std::vector<float> runClassifyGPU(std::vector<float> img);
    std::vector<std::vector<YoloBox>> runYoloGPU(std::vector<float> img);
#endif

#ifdef USE_OPENCL
    std::vector<float> runClassifyCL(std::vector<float> img);
    // std::vector<std::vector<YoloBox>> runYoloCL(std::vector<float> img);

#endif

    Point2I getInputSize();

    int getInputChannel();

    void  clearLayers();

    float getInferenceTime();

    std::string getLayerDetail();

    std::string getTimeDetail();

    float getGpuInferenceTime() const;

    Network *getNet() const;

    int getLastLayerOutWidth() const;

    int getLastLayerOutHeight() const;

    int getLastLayerOutChannel() const;

    size_t getLastLayerOutNum() const;

    void setSaveLayerOutput(bool saveLayerOutput);

    bool getSaveLayerWeights() const;

    void setSaveLayerWeights(bool saveLayerWeights);

private:

    Parser          *_parser;
    Network         *_net;
    NetworkState    *_netState;
    float           _gpuInferenceTime       = 0.f;
    int             _lastLayerOutWidth      = 0;
    int             _lastLayerOutHeight     = 0;
    int             _lastLayerOutChannel    = 0;
    size_t          _lastLayerOutNum        = 0;
    bool            _saveLayerOutput        = false;
    bool            _saveLayerWeights       = false;
};
}
#endif 

