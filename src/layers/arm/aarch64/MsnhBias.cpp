#ifdef USE_ARM
#include "Msnhnet/layers/arm/aarch64/MsnhBias.h"
namespace Msnhnet
{
void BiasLayerArm::BiasV8(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &bias, float* dest)
{
    const int inSize = inHeight * inWidth;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int cc = 0; cc < inChannel; cc++){
        float *srcptr = src + cc * inSize;
        float *destptr = dest + cc * inSize;
        float Bias = *(bias + cc);
#if USE_ARM
        int nn = inSize >> 2;
        int remain = inSize - (nn << 2);
#else
        int remain = inSize;
#endif

#if USE_ARM
#if __aarch64__
        if(nn > 0){
            asm volatile(
                "dup        v0.4s, %w6          \n"

                "0:                             \n"
                "prfm       pldl1keep, [%0, #128]   \n"
                "ld1        {v1.4s}, [%0], #16  \n"
                "fadd       v1.4s, v1.4s, v0.4s \n"
                "st1        {v1.4s}, [%1], #16  \n"

                "subs       %w2, %w2, #1        \n"
                "bne        0b                  \n"

                : "=r"(srcptr), // %0
                "=r"(destptr),  // %1
                "=r"(nn)        // %2
                : "0"(srcptr),
                "1"(destptr),   
                "2"(nn),
                "r"(Bias)         // %w6
                : "cc", "memory", "v0", "v1"
            );
        }
#else
        if(nn > 0){
            asm volatile(
                "vdup.f32   q1, %6              \n"

                "0:                             \n"
                "vld1.f32   {d0-d1}, [%1]!      \n"
                "vadd.f32   q0, q1, q0          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn), // %0
                "=r"(srcptr), // %1
                "=r"(destptr)
                : "0"(nn),
                "1"(srcptr),
                "2"(destptr),
                "r"(Bias)
                : "cc", "memory", "q0"
            );
        }
#endif

#endif
        for(; remain > 0; remain--){
            (*destptr) = (*srcptr) + Bias;
            srcptr++;
            destptr++;
        }

    }
}

void BiasLayerArm::BiasInplaceV8(float* src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &bias)
{
    const int inSize = inHeight * inWidth;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int cc = 0; cc < inChannel; cc++){
        float *srcptr = src + cc * inSize;
        float Bias = *(bias + cc);
#if USE_ARM
        int nn = inSize >> 2;
        int remain = inSize - (nn << 2);
#else
        int remain = inSize;
#endif

#if USE_ARM

        if(nn > 0){
            asm volatile(
                "dup        v0.4s, %w4          \n"

                "0:                             \n"
                "prfm       pldl1keep, [%0, #128]   \n"
                "ld1        {v1.4s}, [%0]       \n"
                "fadd       v1.4s, v1.4s, v0.4s \n"
                "st1        {v1.4s}, [%0], #16  \n"

                "subs       %w1, %w1, #1        \n"
                "bne        0b                  \n"

                : "=r"(srcptr), // %0
                "=r"(nn)        // %1
                : "0"(srcptr),
                "1"(nn),
                "r"(Bias)          // %w4
                : "cc", "memory", "v0", "v1"
            );
        }


#endif
        for(; remain > 0; remain--){
            (*srcptr) = (*srcptr) + Bias;
            srcptr++;
        }

    }

}

#endif
