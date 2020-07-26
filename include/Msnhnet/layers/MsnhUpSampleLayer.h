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
    UpSampleLayer(const int &_batch, const int &_width, const int &_height, const int &_channel, const int &_stride, const float &_scale);

    virtual void forward(NetworkState &netState);
    void resize(const int &_width, const int &_height);

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

