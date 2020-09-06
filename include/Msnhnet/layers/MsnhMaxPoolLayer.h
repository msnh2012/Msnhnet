#ifndef MSNHMAXPOOLLAYER_H
#define MSNHMAXPOOLLAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhMaxPoolLayerGPU.h"
#endif

namespace Msnhnet
{
class MsnhNet_API MaxPoolLayer:public BaseLayer
{
public:
    MaxPoolLayer(const int &batch,  const int &height, const int &width, const int &channel, const int &kSizeX, const int &kSizeY,
                 const int &strideX, const int &strideY, const int &paddingX, const int &paddingY, const int &maxPoolDepth,
                 const int &outChannelsMp, const int &ceilMode, const int &antialiasing);

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

#ifdef USE_X86
    void forwardAvx(float *const &src, float *const &dst, const int &kSizeX, const int &kSizeY, const int &width,
                    const int &height, const int &outWidth, const int &outHeight, const int &channel, const int &paddingX,
                    const int &paddingY, const int &stride, const int &batch);
#endif

    ~MaxPoolLayer();

    int getKSizeX() const;

    int getKSizeY() const;

    int getStride() const;

    int getStrideX() const;

    int getStrideY() const;

    int getPaddingX() const;

    int getPaddingY() const;

    int getAntialiasing() const;

    int getMaxPoolDepth() const;

    int getOutChannelsMp() const;

    int getCeilMode() const;

protected:
    int         _kSizeX              =   0;
    int         _kSizeY              =   0;
    int         _stride              =   0;
    int         _strideX             =   0;
    int         _strideY             =   0;
    int         _paddingX            =   0;
    int         _paddingY            =   0;
    int         _antialiasing        =   0;
    int         _maxPoolDepth        =   0;
    int         _outChannelsMp       =   0;
    int         _ceilMode            =   0;

#ifdef USE_GPU
#ifdef USE_CUDNN

    cudnnPoolingDescriptor_t        _maxPoolDesc;

    cudnnTensorDescriptor_t         _inputDesc;
    cudnnTensorDescriptor_t         _outputDesc;
#endif
#endif
};
}

#endif 

