#ifdef USE_ARM
#include "Msnhnet/layers/arm/aarch64/MsnhBatchNorm.h"
namespace Msnhnet
{
void BatchNormLayerArm::BatchNormV8(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float* dest,
                                  float *const &Scales, float *const &rollMean, float *const &rollVariance, float *const &biases, const float &eps )
{
    const int in_size = inWidth * inHeight;
    const float *srcPtr0 = src ;
    float *destPtr0 = dest;
#if USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int i = 0; i < inChannel; i++){

        const float *srcPtr = src + i * in_size;
        float *destPtr = dest + i * in_size;

        float sqrtVar = sqrt(rollVariance[i] + eps);
        float a = biases[i] - Scales[i] * rollMean[i] / sqrtVar;
        float b = Scales[i] / sqrtVar;

#if USE_ARM
        int nn = in_size >> 2;
        int remain = in_size - (nn << 2);
#else
        int remain = in_size;
#endif

#if USE_ARM
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
                "dup        v0.4s, %w6          \n"
                "dup        v1.4s, %w7          \n"

                "0:                             \n"
                "prfm       pldl1keep, [%0, #128]   \n"
                "ld1        {v2.4s}, [%0], #16  \n"
                "fmul       v2.4s, v2.4s, v1.4s \n"
                "fadd       v2.4s, v2.4s, v0.4s \n"
                "st1        {v2.4s}, [%1], #16  \n"

                "subs       %w2, %w2, #1             \n"
                "bne        0b                  \n"

                : "=r"(srcPtr), // %0
                "=r"(destPtr),  // %1
                "=r"(nn)        // %2
                : "0"(srcPtr),
                "1"(destPtr),   
                "2"(nn),
                "r"(a),         // %w6
                "r"(b)          // %w7
                : "cc", "memory", "v0", "v1", "v2"
            );
        }

        for(; remain > 0; remain--){
            *destPtr = b * (*srcPtr) + a;
            srcPtr++;
            destPtr++;
        }
#else
        for(int j=0; j<remain; j++){
            *(destPtr0 + i*remain + j) = b * (*(srcPtr0 + i*remain + j)) + a;
        }

#endif

    }
}

void BatchNormLayerArm::BatchNormInplaceV8(float* src, const int &inWidth, const int &inHeight,  const int &inChannel,
                                  float *const &Scales, float *const &rollMean, float *const &rollVariance, float *const &biases, const float &eps )
{

    const int in_size = inWidth * inHeight;
    float *srcPtr0 = src ;
#if USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int i = 0; i < inChannel; i++){

        float *srcPtr = src + i * in_size;

        float sqrtVar = sqrt(rollVariance[i] + eps);
        float a = biases[i] - Scales[i] * rollMean[i] / sqrtVar;
        float b = Scales[i] / sqrtVar;

#if USE_ARM
        int nn = in_size >> 2;
        int remain = in_size - (nn << 2);
#else
        int remain = in_size;
#endif

#if USE_ARM
        if(nn > 0){

            asm volatile(
                "dup        v0.4s, %w4          \n"
                "dup        v1.4s, %w5          \n"

                "0:                             \n"
                "prfm       pldl1keep, [%0, #128]   \n"
                "ld1        {v2.4s}, [%0]       \n"
                "fmul       v2.4s, v2.4s, v1.4s \n"
                "fadd       v2.4s, v2.4s, v0.4s \n"
                "st1        {v2.4s}, [%0], #16  \n"

                "subs       %w1, %w1, #1        \n"
                "bne        0b                  \n"

                : "=r"(srcPtr), // %0
                "=r"(nn)        // %1
                : "0"(srcPtr),
                "1"(nn),
                "r"(a),         // %w4
                "r"(b)          // %w5
                : "cc", "memory", "v0", "v1", "v2"
            );

        }

        for(; remain > 0; remain--){
            *srcPtr = b * (*srcPtr) + a;
            srcPtr++;
        }
#else
        for(int j=0; j<remain; j++){
            *(srcPtr0 + i*remain + j) = b * (*(srcPtr0 + i*remain + j)) + a;
        }

#endif

    }

}

}

#endif
