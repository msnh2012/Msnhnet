#ifndef MSNHGLOBALAVGPOOL_H
#define MSNHGLOBALAVGPOOL_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhGlobalAvgPoolLayerGPU.h"
#endif

namespace Msnhnet
{

class MsnhNet_API GlobalAvgPoolLayer: public BaseLayer
{
public:
    GlobalAvgPoolLayer(const int &batch, const int &height, const int &width, const int &channel);
    ~GlobalAvgPoolLayer();

    virtual void mallocMemory();
#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    virtual void forward(NetworkState &netState);
};

}

#endif 

