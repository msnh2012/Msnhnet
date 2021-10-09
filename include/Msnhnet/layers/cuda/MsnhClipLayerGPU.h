#ifndef MSNHCLIPLAYERGPU_H
#define MSNHCLIPLAYERGPU_H
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{
class MsnhNet_API ClipLayerGPU
{
public:
    static void forwardNormalGPU(const int &batch,  const int &outChannel, const int &outHeight, const int &outWidth,
                                   const float &min,  const float &max,
                                   float *const &input, float * const &output);
};

}
#endif 

