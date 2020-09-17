#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhAbsVal.h"
namespace Msnhnet
{
void AbsValLayerArm::AbsVal(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float* dest)
{
    const int inSize = inHeight * inWidth;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int cc = 0; cc < inChannel; cc++){
        float *srcptr = src + cc * inSize;
        float *destptr = dest + cc * inSize;
#if USE_NEON
        int nn = inSize >> 2;
        int remain = inSize - (nn << 2);
#else
        int remain = inSize;
#endif

#if USE_NEON
#if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
        if(nn > 0){
            asm volatile(
                "0:                             \n"
                "vld1.f32   {d0-d1}, [%1]!      \n"
                "vabs.f32   q0, q0              \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn), // %0
                "=r"(srcptr), // %1
                "=r"(destptr)
                : "0"(nn),
                "1"(srcptr),
                "2"(destptr)
                : "cc", "memory", "q0"
            );
        }
#endif

#endif
        for(; remain > 0; remain--){
            (*destptr) = (*srcptr) > 0 ? (*srcptr) : -(*srcptr);
            srcptr++;
            destptr++;
        }

    }
}

void AbsValLayerArm::AbsValInplace(float* src, const int &inWidth, const int &inHeight,  const int &inChannel)
{
    const int inSize = inHeight * inWidth;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int cc = 0; cc < inChannel; cc++){
        float *srcptr = src + cc * inSize;
#if USE_NEON
        int nn = inSize >> 2;
        int remain = inSize - (nn << 2);
#else
        int remain = inSize;
#endif

#if USE_NEON
#if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
        if(nn > 0){
            asm volatile(
                "0:                             \n"
                "vld1.f32   {d0-d1}, [%1]       \n"
                "vabs.f32   q0, q0              \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1]!      \n"
                "bne        0b                  \n"
                : "=r"(nn), // %0
                "=r"(srcptr) // %1
                : "0"(nn),
                "1"(srcptr)
                : "cc", "memory", "q0"
            );
        }
#endif

#endif
        for(; remain > 0; remain--){
            (*srcptr) = (*srcptr) > 0 ? (*srcptr) : -(*srcptr);
            srcptr++;
        }

    }

}

#endif
