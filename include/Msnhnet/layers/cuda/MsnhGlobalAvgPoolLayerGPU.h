#ifndef MSNHGLOBALAVGPOOLLAYERGPU_H
#define MSNHGLOBALAVGPOOLLAYERGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{

class GlobalAvgPoolLayerGPU
{
public:
    static void forwardNormalGPU(const int &width, const int &height, const int &channel, const int &batch, float *const &input, float *const &output);
};

}

#endif
