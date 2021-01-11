#ifndef MSNHBLASCHW4_H
#define MSNHBLASCHW4_H
#include <algorithm>
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhSimd.h"

namespace Msnhnet
{
#define NCHW4_PACK 4
class MsnhNet_API BlasNCHW4
{
public:
    static void cpuNCHWToNCHW4(float *const &org, const int width, const int height, const int channel, const int batch, float * const &dstNCHW4);

    static void cpuNCHW4ToNCHW(float *const &orgNCHW4, const int width, const int height, const int channel, const int outChannel, const int batch, float * const &dst);

    static size_t getNCHW4Params(const int width, const int height, const int channel, int &outWidth, int& outChannel);

    static void cpuFillNCHW4(const int &inputN, const float &alpha, float *const &xNCHW4);

    static void cpuAxpyNCHW4(const int &inputN, const float &alpha, float *const &xNCHW4,float *const &yNCHW4);

    static void cpuArithmeticNCHW4(const Arithmetic &type, const int &inputN, float *const &xNCHW4, float *const &yNCHW4, float *outNCHW4);

    static void cpuArithmeticNCHW4(const Arithmetic &type, const int &inputN, float *const &xNCHW4, const float alpha, float *outNCHW4);

    static void cpuScientificNCHW4(const Scientific &type, const int &inputN, float *const &xNCHW4, const float alpha, float *outNCHW4);

    static void cpuNormNCHW4(float *const &xNCHW4, float *const &meanNCHW4, float *const &varNCHW4, const int &batch,
                              const int &filtersNCHW4, const float &eps, const int &whSize);

    static void cpuUpSampleNCHW4(float *const &inNCHW4, const int &width, const int &height, const int &channelNCHW4, const int &batch,
                                   const int &strideX, const int &strideY, const float &scale, float *const &outNCHW4);

    static void cpuSoftmaxNCHW4(float *const &inputNCHW4, const int &num, const int &batch, const int &batchOff,
                                   const int &groups, const int &groupOff, const float &temperature,  const int &stride,
                                   float *const &outputNCHW4);

    static void cpuBilinearResizeNCHW4(float *const &inNCHW4, const int &width, const int &height, const int &channelNCHW4, const int &batch, const int &outWidth,
                                          const int &outHeight, const int &alignCorners, float *const &outNCHW4);
};

}

#endif 

