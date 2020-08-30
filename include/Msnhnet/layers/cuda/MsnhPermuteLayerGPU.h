#ifndef MSNHPERMUTELAYERGPU_H
#define MSNHPERMUTELAYERGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{
class MsnhNet_API PermuteLayerGPU
{
public:
    static void forwardNormalGPU(const int &batch, const int &outChannel, const int &outHeight, const int &outWidth,
                                   const int &height, const int &width, const int &channel,
                                   const int &dim0, const int &dim1, const int &dim2,
                                   float *const &input, float *const &output);
};

}
#endif 

