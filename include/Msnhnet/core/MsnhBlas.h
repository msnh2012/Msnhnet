#ifndef MSNHBLAS_H
#define MSNHBLAS_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhSimd.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API Blas
{
public:
    Blas();

    static void cpuCopy(const int &inputN, float *const &input, const int &inputStep,
                        float *const &output, const int &outputStep);

    static void cpuFill(const int &inputN, const float &alpha, float *const &x, const int &step);

    static void cpuAxpy(const int &inputN, const float &alpha, float *const &x,
                        const int &stepX, float *const &y, const int &stepY);

    static void cpuScale(const int &inputN, const float &alpha, float *const &x, const int &stepX);

    static void cpuMean(float *const &x, const int &batch, const int &filters, const int &outSize, float *const &mean);

    static void cpuVariance(float *const &x, float *const &mean, const int &batch,
                            const int &filters, const int &outSize, float *const &variance);

    static void cpuNorm(float *const &x, float *const &mean, float *const &variance,
                        const int &batch, const int &filters, const int &outSize);

    static void cpuSmoothL1(const int &n, float *const &pred, float *const &truth, float *const &delta, float *const &error);

    static void cpuL1(const int &n, float *const &pred, float *const &truth, float *const &delta, float *const &error);

    static void cpuL2(const int &n, float *const &pred, float *const &truth, float *const &delta, float *const &error);

    static void cpuFlatten(float *const &x, const int &size, const int &layers, const int &batch, const int &forward);

    static void softmax(float *const &input, const int &num, const float &temp, const int &stride, float *const &output);

    static void cpuSoftmax(float *const &input, const int &num, const int &batch, const int &batchOff,
                           const int &groups, const int &groupOff, const float &temperature,  const int &stride,
                           float *const &output);

    static void cpuSoftMaxCrossEntropy(const int &num, float *const &pred, float *const &truth, float *const &delta, float *const &error);

    static void cpuLogisticCorssEntropy(const int &num, float *const &pred, float *const &truth, float *const &delta, float *const &error);

    static void cpuUpSample(float *const &in, const int &width, const int &height, const int &channel, const int &batch, const int &stride,
                            const int &forward, const float &scale, float *const &out);

    static inline float relu(const float &x)
    {
        if(x > 0)
        {
            return x;
        }
        else
        {
            return 0;
        }
    }
};
}
#endif 
