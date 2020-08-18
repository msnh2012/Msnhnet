#ifndef MSNHUPSAMPLE_H
#define MSNHUPSAMPLE_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API UpSampleLayer : public BaseLayer
{
public:
    UpSampleLayer(const int &batch, const int &width, const int &height, const int &channel, const int &stride, const float &scale);

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif
    void resize(const int &width, const int &height);

    int getReverse() const;

    int getStride() const;

    float getScale() const;

protected:
    int         _reverse     =   0;
    int         _stride      =   0;
    float       _scale       =   1.f;
};
}

#endif 

