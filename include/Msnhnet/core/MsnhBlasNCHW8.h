#ifndef MSNHBLASNCHW8_H
#define MSNHBLASNCHW8_H

#ifdef USE_X86

#include <algorithm>
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhSimd.h"

namespace Msnhnet
{
#define NCHW8_PACK 8
class MsnhNet_API BlasNCHW8
{
public:
    static void cpuNCHWToNCHW8(float *const &org, const int width, const int height, const int channel, const int batch, float * const &dstNCHW8);

    static void cpuNCHW8ToNCHW(float *const &orgNCHW8, const int width, const int height, const int channel, const int outChannel, const int batch, float * const &dst);

    static size_t getNCHW8Params(const int width, const int height, const int channel, int &outWidth, int& outChannel);

    static void cpuFillNCHW8(const int &inputN, const float &alpha, float *const &xNCHW8);

    static void cpuAxpyNCHW8(const int &inputN, const float &alpha, float * const &xNCHW8, float * const &yNCHW8);

    static void cpuArithmeticNCHW8(const Arithmetic &type, const int &inputN, float *const &xNCHW8, float *const &yNCHW8, float *outNCHW8);

    static void cpuArithmeticNCHW8(const Arithmetic &type, const int &inputN, float *const &xNCHW8, const float alpha, float *outNCHW8);

    static void cpuScientificNCHW8(const Scientific &type, const int &inputN, float *const &xNCHW8, const float alpha, float *outNCHW8);

    static void cpuNormNCHW8(float *const &xNCHW8, float *const &meanNCHW8, float *const &varNCHW8, const int &batch,
                             const int &filtersNCHW8, const float &eps, const int &whSize);

    static void cpuUpSampleNCHW8(float *const &inNCHW8, const int &width, const int &height, const int &channelNCHW8, const int &batch,
                                   const int &strideX, const int &strideY, const float &scale, float *const &outNCHW8);

    static void cpuSoftmaxNCHW8(float *const &input, const int &num, const int &batch, const int &batchOff,
                                   const int &groups, const int &groupOff, const float &temperature,  const int &stride,
                                   float *const &output);

    static void cpuBilinearResizeNCHW8(float *const &inNCHW8, const int &width, const int &height, const int &channelNCHW8, const int &batch, const int &outWidth,
                                          const int &outHeight, const int &alignCorners, float *const &outNCHW8);
};

}

#endif
#endif 

