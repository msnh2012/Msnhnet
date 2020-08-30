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

            const float *srcPtr = src + i * in_size;
            float *destPtr = dest + i * in_size;

            float sqrtVar = sqrt(rollVariance[i] + 0.00001f);
            float a = biases[i] - Scales[i] * rollMean[i] / sqrtVar;
            float b = Scales[i] / sqrtVar;

#if USE_NEON
        nn = in_size >> 2;
        remain = in_size - (nn << 2);
        //float32x4_t a_new = vdupq_n_f32(a);
        //float32x4_t b_new = vdupq_n_f32(b);
#else
        remain = in_size;
#endif

#if USE_NEON
        // for(; nn > 0; nn--){
        //     #if __aarch64__
        //         throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
        //     #else
        //         float32x4_t tmp = vld1q_f32(srcPtr);
        //         float32x4_t sum = vmulq_f32(tmp, b_new);
        //         sum = vaddq_f32(sum, a_new);
        //         vst1q_f32(destPtr, sum);
        //     #endif
        //     srcPtr += 4;
        //     destPtr += 4;
        // }

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

        for(; remain > 0; remain--){
            *destPtr = b * (*srcPtr) + a;
            srcPtr++;
            destPtr++;
        }

#else
        for(; remain > 0; remain--){
            *destPtr = b * (*srcPtr) + a;
            srcPtr++;
            destPtr++;
        }

#endif
    }
}

}

#endif
