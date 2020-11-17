#ifndef MSNHACTIVATIONSGPU_H
#define MSNHACTIVATIONSGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{

class MsnhNet_API ActivationsGPU
{
public:
static void gpuActivateArray(float *const &gpuX, const int &numX, const ActivationType &actType, const float &param=0.1f);
static void gpuActivatePRelu(float *const &gpuX, const int &batch, const int &channels, float *const &gpuWeights, const int &whStep);
static void gpuActivateArrayNormCh(float *const &gpuX, const int &numX, const int &batch, const int &channels, const int &whStep, float *const &gpuOutput);
static void gpuActivateArrayNormChSoftMax(float *const &gpuX, const int &numX, const int &batch, const int &channels, const int &whStep, float *const &gpuOutput, const int &useMaxVal);
};

}

#endif
