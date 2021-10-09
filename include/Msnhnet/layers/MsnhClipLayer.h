#ifndef MSNHCLIPLAYER_H
#define MSNHCLIPLAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API ClipLayer: public BaseLayer
{
public:
    ClipLayer(const int& batch, const int& height, const int &width, const int &channel, const float& min, const float &max);

    virtual void mallocMemory();

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    void forwardGPU(NetworkState &netState);
#endif

    float getMax() const;

    float getMin() const;

private:
    float _max;
    float _min;
};
}

#endif 

