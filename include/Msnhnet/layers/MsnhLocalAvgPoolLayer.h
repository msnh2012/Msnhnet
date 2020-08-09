#ifndef MSNHLOCALAVGPOOL_H
#define MSNHLOCALAVGPOOL_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhLocalAvgPoolLayerGPU.h"
#endif

namespace Msnhnet
{
class MsnhNet_API LocalAvgPoolLayer : public BaseLayer
{
public:
    LocalAvgPoolLayer(const int &_batch, const int &_height, const int &_width, const int &_channel, const int &_kSizeX, const int &_kSizeY,
                      const int &_strideX, const int &_strideY, const int &_paddingX, const int &_paddingY, const int&_ceilMode, const int &_antialiasing);

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    ~LocalAvgPoolLayer();

    int getKSizeX() const;

    int getKSizeY() const;

    int getStride() const;

    int getStrideX() const;

    int getStrideY() const;

    int getPaddingX() const;

    int getPaddingY() const;

    int getCeilMode() const;

    int getAntialiasing() const;

protected:
    int         _kSizeX              =   0;
    int         _kSizeY              =   0;
    int         _stride              =   0;
    int         _strideX             =   0;
    int         _strideY             =   0;
    int         _paddingX            =   0;
    int         _paddingY            =   0;

    int         _ceilMode            =   0;

    int         _antialiasing        =   0;
#ifdef USE_GPU
#ifdef USE_CUDNN

    cudnnPoolingDescriptor_t        _localAvgPoolDesc;

    cudnnTensorDescriptor_t         _inputDesc;
    cudnnTensorDescriptor_t         _outputDesc;
#endif
#endif
};
}

#endif 

