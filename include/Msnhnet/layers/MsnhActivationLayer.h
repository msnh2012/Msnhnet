#ifndef ACTIVATIONLAYER_H
#define ACTIVATIONLAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/layers/MsnhActivations.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API ActivationLayer:public BaseLayer
{
public:
    ActivationLayer(const int &batch, const int &width, const int &height, const int &channel, const int &inputNum, const ActivationType &activation);
    virtual void forward(NetworkState &netState);
#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif
};
}

#endif 

