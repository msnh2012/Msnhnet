#ifndef MSNHCONVOLUTIONLAYERGPU_H
#define MSNHCONVOLUTIONLAYERGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{

class MsnhNet_API ConvolutionalLayerGPU
{
public:
    static void convBn(const int &batch, const int &outChannel, const int &outHeight, const int &outWidth, float* const &gpuScales,
                       float *const &gpuRollMean, float *const &gpuRollVariance, float *const &gpuBiases, float *const &gpuOutput);
};

}

#endif
