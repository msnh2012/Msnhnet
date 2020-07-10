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
    MaxPoolLayer(const int &batch,  const int &height, const int &width, const int &channel, const int &kSizeX, const int &kSizeY,
                 const int &strideX, const int &strideY, const int &paddingX, const int &paddingY, const int &maxPoolDepth,
                 const int &outChannelsMp, const int &ceilMode, const int &antialiasing);

    int         kSizeX              =   0;
    int         kSizeY              =   0;
    int         stride              =   0;
    int         strideX             =   0;
    int         strideY             =   0;
    int         paddingX            =   0;
    int         paddingY            =   0;
    int         antialiasing        =   0;
    int         maxPoolDepth        =   0;
    int         outChannelsMp       =   0;
    int         ceilMode            =   0;

    virtual void forward(NetworkState &netState);

#ifdef USE_X86
    void forwardAvx(float *const &src, float *const &dst, const int &kSizeX, const int &kSizeY, const int &width,
                    const int &height, const int &outWidth, const int &outHeight, const int &channel, const int &paddingX,
                    const int &paddingY, const int &stride, const int &batch);
#endif

    ~MaxPoolLayer();

};
}

#endif 

