#ifndef MSNHEMPTYLAYER_H
#define MSNHEMPTYLAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API EmptyLayer : public BaseLayer
{
public:
    EmptyLayer(const int &_batch, const int &_width, const int &_height, const int &_channel);
    ~EmptyLayer();

    virtual  void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif
};
}

#endif 

