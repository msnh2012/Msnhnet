#ifndef MSNHYOLOV3LAYERGPU_H
#define MSNHYOLOV3LAYERGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{

class MsnhNet_API Yolov3LayerGPU
{
public:
    static void exSigmoidGpu(const int &n, float *const &input, const int &width, const float &ratios, const int &addGrid);

    static void sigmoidGpu(const int &n, float *const &input);

    static void aExpTGpu(const int &n, float *const &input, const float &a);
};

}

#endif
