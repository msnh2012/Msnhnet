#ifndef MSNHUPSAMPLE_H
#define MSNHUPSAMPLE_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/io/MsnhParser.h"

namespace Msnhnet
{
class MsnhNet_API UpSampleLayer : public BaseLayer
{
public:
    UpSampleLayer(const int &batch, const int &width, const int &height, const int &channel, const int &strideX, const int &strideY, const float &scaleX, const float &scaleY, UpSampleParams::UpsampleType upsampleType, const int &alignCorners);

    virtual void forward(NetworkState &netState);

    virtual void mallocMemory();

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif
    void resize(const int &width, const int &height);

    int getStrideX() const;

    int getStrideY() const;

    float getScaleX() const;

    float getScaleY() const;

    int getAlignCorners() const;

    UpSampleParams::UpsampleType getUpsampleType() const;

protected:
    int         _strideX     =   1;
    int         _strideY     =   1;
    float       _scaleX      =   1.f;
    float       _scaleY      =   1.0f;
    int         _alignCorners=   0;
    UpSampleParams::UpsampleType _upsampleType;
};
}

#endif 

