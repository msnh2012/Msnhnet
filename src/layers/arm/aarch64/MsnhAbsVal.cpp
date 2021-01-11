#ifdef USE_ARM
#include "Msnhnet/layers/arm/aarch64/MsnhAbsVal.h"
namespace Msnhnet
{
void AbsValLayerArm::AbsValV8(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float* dest)
{
    const int inSize = inHeight * inWidth;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int cc = 0; cc < inChannel; cc++){
        float *srcptr = src + cc * inSize;
        float *destptr = dest + cc * inSize;
#if USE_ARM
        int nn = inSize >> 2;
        int remain = inSize - (nn << 2);
#else
        int remain = inSize;
#endif

#if USE_ARM

        if(nn > 0){
            asm volatile(
                "0:                             \n"
                "prfm       pldl1keep, [%0, #128]   \n"
                "ld1        {v0.4s}, [%0], #16  \n"
                "fabs       v0.4s, v0.4s        \n"
                "st1        {v0.4s}, [%1], #16  \n"
                "subs       %w2, %w2, #1        \n"
                "bne        0b                  \n"
                : "=r"(srcptr),
                "=r"(destptr),
                "=r"(nn)
                : "0"(srcptr),      // %0
                "1"(destptr),       // %1
                "2"(nn)             // %w2
                : "cc", "memory", "v0"
            );
        }

#endif
        for(; remain > 0; remain--){
            (*destptr) = (*srcptr) > 0 ? (*srcptr) : -(*srcptr);
            srcptr++;
            destptr++;
        }

    }
}

void AbsValLayerArm::AbsValInplaceV8(float* src, const int &inWidth, const int &inHeight,  const int &inChannel)
{
    const int inSize = inHeight * inWidth;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int cc = 0; cc < inChannel; cc++){
        float *srcptr = src + cc * inSize;
#if USE_ARM
        int nn = inSize >> 2;
        int remain = inSize - (nn << 2);
#else
        int remain = inSize;
#endif

#if USE_ARM

        if(nn > 0){
            asm volatile(
                "0:                             \n"
                "prfm       pldl1keep, [%0, #128]   \n"
                "ld1        {v0.4s}, [%0]       \n"
                "fabs       v0.4s, v0.4s        \n"
                "st1        {v0.4s}, [%0], #16  \n"
                "subs       %w1, %w1, #1        \n"
                "bne        0b                  \n"
                : "=r"(srcptr),
                "=r" (nn)
                : "0"(srcptr),      // %0
                "1"(nn)             // %w1
                : "cc", "memory", "v0"
            );
        }


#endif
        for(; remain > 0; remain--){
            (*srcptr) = (*srcptr) > 0 ? (*srcptr) : -(*srcptr);
            srcptr++;
        }

    }

}

#endif
