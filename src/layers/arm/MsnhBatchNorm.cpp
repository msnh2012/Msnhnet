#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhBatchNorm.h"
namespace Msnhnet
{
void BatchNormLayerArm::BatchNorm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float* dest,
                                  float *const &Scales, float *const &rollMean, float *const &rollVariance, float *const &biases)
{
    int in_size = inWidth * inHeight;
    const float *srcPtr = src;
    float *destPtr = dest;
    int nn, remain;
#if USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int i = 0; i < inChannel; i++){

        float sqrtVar = sqrt(rollVariance[i] + 0.00001f);
        float a = biases[i] - Scales[i] * rollMean[i] / sqrtVar;
        float b = Scales[i] / sqrtVar;

#if USE_NEON
        nn = in_size >> 2;
        remain = in_size - (nn << 2);
#else
        remain = in_size;
#endif

#if USE_NEON

        if(nn > 0){
            asm volatile(
                        "vdup.f32   q0, %6              \n"
                        "vdup.f32   q1, %7              \n"

                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d4,d5}, [%1]!      \n"

                        "vmul.f32   q3, q2, q1          \n"
                        "vadd.f32   q4, q3, q0          \n"

                        "vst1.f32   {d8-d9}, [%2]!      \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"


                        : "=r"(nn),     // %0
                        "=r"(srcPtr), // %1
                        "=r"(destPtr)     // %2
                        : "0"(nn),
                        "1"(srcPtr), //
                        "2"(destPtr),
                        "r"(a), // %6
                        "r"(b) // %7
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4"
                        );
        }

#endif
        for(; remain > 0; remain--){
            *destPtr = b * (*srcPtr) + a;
            srcPtr++;
            destPtr++;
        }

    }
}

}

#endif