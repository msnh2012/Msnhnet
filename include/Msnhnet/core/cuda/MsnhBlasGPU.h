#ifndef MSNHBLASGPU_H
#define MSNHBLASGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{

class MsnhNet_API BlasGPU
{
public:
    BlasGPU();

    static void gpuSimpleCopy(const int &size,  float * const &src, float * const &dst);

    static void gpuCopy(const int &n, float * const x, const int &incX, float * const y, const int &incY);

    static void gpuMemcpy(void *const &dst, void *const &src, const int &size);

    static void gpuFill(const int &n, const float &alpha, float *const &x, const int &step);

    static void gpuAxpy(const int &n, const float &alpha, float * const x, const int &incX, float * const y, const int &incY);

    static void gpuArithmetic(const Arithmetic &type, const int &n, float * const &x, const int &stepX, float * const &y, const int &stepY,
                              float *out, const int &stepOut);

    static void gpuArithmetic(const Arithmetic &type, const int &n, float *const &x, const int &stepX,
                               const float &alpha, float *out, const int &stepOut);

    static void gpuScale(const int &n, const float &alpha, float *const &x, const int &step);

    static void gpuMean(float *const &x, const int &batch, const int &filters, const int &outSize, float *const &mean);

    static void gpuVariance(float *const &x, float *const &mean, const int &batch,
                            const int &filters, const int &outSize, float *const &variance);

    static void gpuNorm(float *const &x, float *const &mean, float *const &variance,
                        const int &batch, const int &filters, const int &outSize);

    static void gpuSmoothL1(const int &n, float *const &pred, float *const &truth, float *const &delta, float *const &error);

    static void gpuL1(const int &n, float *const &pred, float *const &truth, float *const &delta, float *const &error);

    static void gpuL2(const int &n, float *const &pred, float *const &truth, float *const &delta, float *const &error);

    static void gpuSoftmax(float *const &input, const int &num, const int &batch, const int &batchOff,const int &groups,
                           const int &groupOff, const float &temperature,  const int &stride, float *const &output);

    static void gpuSoftMaxCrossEntropy(const int &num, float *const &pred, float *const &truth, float *const &delta, float *const &error);

    static void gpuLogisticCorssEntropy(const int &num, float *const &pred, float *const &truth, float *const &delta, float *const &error);

    static void gpuUpSample(float *const &in, const int &width, const int &height, const int &channel, const int &batch, const int &stride,
                            const int &forward, const float &scale, float *const &out);

    static void gpuAddBias(float *const &output, float *const &biases, const int &batch, const int &num, const int &whSize);
    static void gpuScaleBias(float *const &output, float *const &scales, const int &batch, const int &num, const int &whSize);
    static void gpuFixNanAndInf(float *const &input, const size_t &size);

};

}
#endif
