#ifndef MSNHPIXELSHUFFLELAYER_H
#define MSNHPIXELSHUFFLELAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/core/MsnhBlas.h"

#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhPixelShuffleLayerGPU.h"
#endif

namespace Msnhnet
{

class MsnhNet_API PixelShuffleLayer : public BaseLayer
{
public:
    PixelShuffleLayer(const int &batch,  const int &height, const int &width, const int &channel, const int &getFactor);

    ~PixelShuffleLayer();

    virtual void forward(NetworkState &netState);

    virtual void mallocMemory();

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    int getFactor() const;

private:
    int _factor = 0;
};

}
#endif 

