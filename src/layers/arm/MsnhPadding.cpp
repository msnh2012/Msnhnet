#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhPadding.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    void PaddingLayerArm::padding(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel,
                                        float* &dest, const int &top, const int &down, const int &left, const int &right, const int &val){
        
        const int outWidth  = inWidth + left + right;
        const int outHeight = inHeight + top + down;
        const int inSize    = inHeight * inWidth;
        const int outSize   = outHeight * outWidth;
        float pval          = static_cast<float>(val);
        
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int c = 0; c < inChannel; c++){
        // fill top
        int i = 0;
        int nn = 0;
        int remain = 0;

        for(; i < top; i++){
            float *destptr = dest + c * outSize + i * outWidth;

#if USE_NEON
            nn = outWidth >> 2;
            remain = outWidth - (nn << 2);
#else
            remain = outWidth;
#endif

            if(nn > 0){
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    "vdup.f32   q0, %4              \n"

                    "0:                             \n"
                    "vst1.f32   {d0-d1}, [%1]!      \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    : "=r"(nn), // %0
                    "=r"(destptr) // %1
                    : "0"(nn),
                    "1"(destptr),
                    "r"(pval) // %4
                    : "memory", "q0", "q1"
                );
#endif
            }

            for(; remain > 0; remain--){
                *destptr = pval;
                destptr++;
            }

        }

        // fill center

        nn = 0;
        remain = 0;

        for(; i < top + inHeight; i++){
            const float *srcptr = src + c * inSize +  (i - top) * inWidth;
            float *destptr = dest + c * outSize + i * outWidth;
            remain = left;
            for(; remain > 0; remain--){
                *destptr = pval;
                destptr++;
            }

            //memcpy
#if USE_NEON
            nn = inWidth >> 2;
            remain = inWidth - (nn << 2);
#else
            remain = inWidth;
#endif

            if(nn > 0){
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    "0:                             \n"
                    "vld1.f32   {d0-d1}, [%1]       \n"

                    "vst1.f32   {d0-d1}, [%2]!      \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    : "=r"(nn), // %0
                    "=r"(srcptr),
                    "=r"(destptr) // %1
                    : "0"(nn),
                    "1"(srcptr),
                    "2"(destptr)
                    : "memory", "q0", "q1"
                );
#endif
            }

            for(; remain > 0; remain--){
                *destptr = *srcptr;
                srcptr++;
                destptr++;
            }

            remain = right;
            for(; remain > 0; remain--){
                *destptr = pval;
                destptr++;
            }

        }

        nn  = 0;
        remain = 0;
        //fill bottom

        for(; i < outHeight; i++){
            float *destptr = dest + c * outSize +  i * outWidth;

#if USE_NEON
            nn = 0;
            remain = outWidth - (nn << 2);
#else
            remain = outWidth;
#endif

            if(nn > 0){
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    "vdup.f32   q0, %4              \n"
                    "0:                             \n"
                    "vst1.f32   {d0-d1}, [%1]!      \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    : "=r"(nn), // %0
                    "=r"(destptr) // %1
                    : "0"(nn),
                    "1"(destptr),
                    "r"(pval) // %4
                    : "memory", "q0", "q1"
                );
#endif
            }

            for(; remain > 0; remain--){
                *destptr = pval;
                destptr++;
            }

        }
        }
    }
}
#endif


