#ifndef MSNHSLICELAYERGPU_H
#define MSNHSLICELAYERGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{
class MsnhNet_API SliceLayerGPU
{
public:
    static void forwardNormalGPU(const int &batch, const int &outChannel, const int &outHeight, const int &outWidth,
                                   const int &height, const int &width, const int &channel,
                                   const int &start0, const int &step0,
                                   const int &start1, const int &step1,
                                   const int &start2, const int &step2,
                                   float *const &input, float * const &output);
};

}
#endif 

