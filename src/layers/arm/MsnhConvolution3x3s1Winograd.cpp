#define USE_ARM 1
#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolution3x3s1Winograd.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    // kerneltm: [outChannel, inChannel, 8*8]
    //F(m, r) = GgG^T
    void ConvolutionalLayerArm3x3s1Winograd::conv3x3s1WinogradTransformKenel(float *const &kernel, float* &kernel_tm, const int &inChannel, const int &outChannel){
        // 矩阵G
        const float ktm[8][3] = {
            {1.0f, 0.0f, 0.0f},
            {-2.0f / 9, -2.0f / 9, -2.0f / 9},
            {-2.0f / 9, 2.0f / 9, -2.0f / 9},
            {1.0f / 90, 1.0f / 45, 2.0f / 45},
            {1.0f / 90, -1.0f / 45, 2.0f / 45},
            {1.0f / 45, 1.0f / 90, 1.0f / 180},
            {1.0f / 45, -1.0f / 90, 1.0f / 180},
            {0.0f, 0.0f, 1.0f}
        };


#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int outc = 0; outc < outChannel; outc++){
            for(int inc = 0; inc < inChannel; inc++){
                const float* kernel0 = (const float*)kernel + outc * inChannel * 9 + inc * 9;
                float *kernel_tm0 = kernel_tm + outc * inChannel * 64 + inc * 64;

                //需要变换的卷积核
                const float* k0 = kernel0;
                const float* k1 = kernel0 + 3;
                const float* k2 = kernel0 + 6;

                

            }
        }


    }

    void ConvolutionalLayerArm3x3s1Winograd::conv3x3s1WinogradNeon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        return;
    }
}

#endif
