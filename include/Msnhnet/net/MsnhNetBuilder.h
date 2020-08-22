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
#include "Msnhnet/layers/MsnhPermuteLayer.h"
#include "Msnhnet/layers/MsnhSoftMaxLayer.h"
#include "Msnhnet/layers/MsnhUpSampleLayer.h"
#include "Msnhnet/layers/MsnhResBlockLayer.h"
#include "Msnhnet/layers/MsnhRes2BlockLayer.h"
#include "Msnhnet/layers/MsnhAddBlockLayer.h"
#include "Msnhnet/layers/MsnhConcatBlockLayer.h"
#include "Msnhnet/layers/MsnhYolov3Layer.h"
#include "Msnhnet/layers/MsnhYolov3OutLayer.h"
#include "Msnhnet/layers/MsnhPaddingLayer.h"
#include "Msnhnet/io/MsnhIO.h"
#include "Msnhnet/utils/MsnhExport.h"

#ifdef USE_NNPACK
#include <nnpack.h>
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
    void buildNetFromMsnhNet(const std::string &path);
    void loadWeightsFromMsnhBin(const std::string &path);
    void setPreviewMode(const bool &mode);
    void setForceUseCuda(const bool &onlyUseCuda);
    void setUseFp16(const bool &useFp16);
    std::vector<float> runClassify(std::vector<float> img);
    std::vector<std::vector<Yolov3Box>> runYolov3(std::vector<float> img);
#ifdef USE_GPU
    std::vector<float> runClassifyGPU(std::vector<float> img);
    std::vector<std::vector<Yolov3Box>> runYolov3GPU(std::vector<float> img);
#endif

    Point2I getInputSize();
    int getInputChannel();
    void  clearLayers();
    float getInferenceTime();
    std::string getLayerDetail();
    std::string getTimeDetail();
    float getGpuInferenceTime() const;

    Network *getNet() const;

private:

    Parser          *_parser;
    Network         *_net;
    NetworkState    *_netState;
    float           _gpuInferenceTime = 0.f;
};
}
#endif 

