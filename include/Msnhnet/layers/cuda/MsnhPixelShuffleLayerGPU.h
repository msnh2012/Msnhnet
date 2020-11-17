#ifndef MSNHPIXELSHUFFLELAYERGPU_H
#define MSNHPIXELSHUFFLELAYERGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{
class MsnhNet_API PixelShuffleLayerGPU
{
public:
    static void forwardNormalGPU(const int &batch, const int &outChannel, const int &outHeight, const int &outWidth,
                                   const int &height, const int &width, const int &channel,
                                   const int &factor,float *const &input, float *const &output);
};

}
#endif 

