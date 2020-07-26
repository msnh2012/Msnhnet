#ifndef MSNHMAXPOOLLAYER_H
#define MSNHMAXPOOLLAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API MaxPoolLayer:public BaseLayer
{
public:
    MaxPoolLayer(const int &_batch,  const int &_height, const int &_width, const int &_channel, const int &_kSizeX, const int &_kSizeY,
                 const int &_strideX, const int &_strideY, const int &_paddingX, const int &_paddingY, const int &_maxPoolDepth,
                 const int &_outChannelsMp, const int &_ceilMode, const int &_antialiasing);

    virtual void forward(NetworkState &netState);

#ifdef USE_X86
    void forwardAvx(float *const &src, float *const &dst, const int &_kSizeX, const int &_kSizeY, const int &_width,
                    const int &_height, const int &_outWidth, const int &_outHeight, const int &_channel, const int &_paddingX,
                    const int &_paddingY, const int &_stride, const int &_batch);
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

};
}

#endif 

