#ifndef MSNHRES2BLOCKLAYER_H
#define MSNHRES2BLOCKLAYER_H

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

class MsnhNet_API Res2BlockLayer : public BaseLayer
{
public:
    Res2BlockLayer(const int &batch, NetBuildParams &params, std::vector<BaseParams*> &baseParams, std::vector<BaseParams*> &branchParams, ActivationType &activation, const std::vector<float> &actParams);

    std::vector<BaseLayer *> baseLayers;
    std::vector<BaseLayer *> branchLayers;
    float       *activationInput    =   nullptr;

    void loadAllWeigths(std::vector<float> &weights);

    virtual void forward(NetworkState &netState);

    ~Res2BlockLayer();
};
}
#endif 

