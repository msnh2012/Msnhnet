#ifndef MSNHCONVOLUTION3X3LAYERX86_H
#define MSNHCONVOLUTION3X3LAYERX86_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/utils/MsnhTypes.h"

#ifdef USE_X86
namespace Msnhnet
{
class MsnhNet_API Convolution3x3LayerX86
{
public:
    static void convolution3x3S1(float *const &src, const int &height, const int &width, const int &channel,
                                 float *&dst, const int &outHeight, const int &outWidth, const int &outChannel,
                                 float *const &kernel, const bool useFMA = false);

    static void convolution3x3S2(float *const &src, const int &height, const int &width, const int &channel,
                                 float *&dst, const int &outHeight, const int &outWidth, const int &outChannel,
                                 float *const &kernel, const bool useFMA = false);
};
}

#endif

#endif 

