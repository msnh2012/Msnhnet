#ifndef MSNHRESBLOCKLAYER_H
#define MSNHRESBLOCKLAYER_H

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
class MsnhNet_API ResBlockLayer : public BaseLayer
{
public:
    ResBlockLayer(const int &batch, NetBuildParams &params, std::vector<BaseParams*> &baseParams, ActivationType &activation, const std::vector<float> &actParams);
    float       *activationInput    =   nullptr;

   void loadAllWeigths(std::vector<float> &weights);

   std::vector<BaseLayer *> baseLayers;

   virtual void forward(NetworkState &netState);

   ~ResBlockLayer();
};
}
#endif 

