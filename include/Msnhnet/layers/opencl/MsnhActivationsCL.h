#ifndef MSNHACTIVATIONSOPENCL_H
#define MSNHACTIVATIONSOPENCL_H

#ifdef USE_OPENCL
#include <CL/cl.h>
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{

class MsnhNet_API ActivationsCL
{
public:
    static void activateArrayCL(float* X, const int &numX, cl_kernel &kernel, const float &param = 0.1f);
    // static void activatePRelu(float *const &gpuX, const int &batch, const int &channels, float *const &gpuWeights, const int &whStep);
    // static void activateArrayNormCh(float *const &gpuX, const int &numX, const int &batch, const int &channels, const int &whStep, float *const &gpuOutput);
    // static void activateArrayNormChSoftMax(float *const &gpuX, const int &numX, const int &batch, const int &channels, const int &whStep, float *const &gpuOutput, const int &useMaxVal);
};

}
#endif
#endif