#ifndef MSNHCATBLOCKLAYER_H
#define MSNHCATBLOCKLAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhActivationLayer.h"
#include "Msnhnet/layers/MsnhBatchNormLayer.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/layers/MsnhConnectedLayer.h"
#include "Msnhnet/layers/MsnhConvolutionalLayer.h"
#include "Msnhnet/layers/MsnhCropLayer.h"
#include "Msnhnet/layers/MsnhDeConvolutionalLayer.h"
#include "Msnhnet/layers/MsnhLocalAvgPoolLayer.h"
#include "Msnhnet/layers/MsnhMaxPoolLayer.h"
#include "Msnhnet/layers/MsnhRouteLayer.h"
#include "Msnhnet/layers/MsnhSoftMaxLayer.h"
#include "Msnhnet/layers/MsnhUpSampleLayer.h"
#include "Msnhnet/layers/MsnhPaddingLayer.h"
#include "Msnhnet/net/MsnhNetBuilder.h"
#include "Msnhnet/io/MsnhParser.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class NetBuildParams;

class MsnhNet_API ConcatBlockLayer : public BaseLayer
{
public:
    ConcatBlockLayer(const int &batch, NetBuildParams &params, std::vector<std::vector<BaseParams*>> &branchParams, ActivationType &activation, const std::vector<float> &actParams);

    std::vector<std::vector<BaseLayer *>> branchLayers;
    float       *activationInput    =   nullptr;

    void mallocMemory();

    void loadAllWeigths(std::vector<float> &weights);

    virtual void saveWeights(const int &mainIdx, const int &branchIdx=-1, const int &branchIdx1=-1);

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    ~ConcatBlockLayer();
};
}

#endif 

