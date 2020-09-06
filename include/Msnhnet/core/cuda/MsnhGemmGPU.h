#ifndef MSNHGEMMGPU_H
#define MSNHGEMMGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{
class MsnhNet_API GemmGPU
{
public:

    static void gpuGemm(const int &TA,   const int &TB, const int &M, const int &N, const int &K, const float &ALPHA,
                        float *const &A, const int &lda,
                        float *const &B, const int &ldb,
                        const float &BETA,
                        float *const &C, const int &ldc);

    static void gpuIm2ColEx(float *input, const int &channelNum, const int &height, const int &width,
                            const int &kernelH, const int &kernelW, const int &padH, const int &padW,
                            const int &strideH,  const int &strideW, const int &dilationH, const int &dilationW,
                            float *output);

    static void gpuIm2col(float *const &input, const int &channelNum, const int &height, const int &width,
                          const int &kSize,const int &stride, const int &padding, float *const &output);

};

}

#endif 

