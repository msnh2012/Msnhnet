#ifndef MSNHYOLOLAYERGPU_H
#define MSNHYOLOLAYERGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{

class MsnhNet_API YoloLayerGPU
{
public:
    static void exSigmoidGpu(const int &n, float *const &input, const int &width, const float &ratios, const int &addGrid);

    static void exSigmoidV5Gpu(const int &n, float *const &input, const int &width, const float &ratios, const int &addGrid);

    static void sigmoidGpu(const int &n, float *const &input);

    static void aPowSigmoid(const int &n, float *const &input, const float &a);

    static void aExpTGpu(const int &n, float *const &input, const float &a);
};

}

#endif
