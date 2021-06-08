#ifdef USE_ARM
#include "Msnhnet/layers/arm/aarch64/MsnhPadding.h"

namespace Msnhnet
{
    void PaddingLayerArm::paddingV8(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel,
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
            for(int i = 0; i < top; i++){
                float *destptr = dest + c * outSize + i * outWidth;

#if USE_ARM
                int nn = outWidth >> 2;
                int remain = outWidth - (nn << 2);
#else
                int remain = outWidth;
#endif

#if USE_ARM

                if(nn > 0){
                    asm volatile(
                        "dup        v0.4s, %w4          \n"
                        "0:                             \n"
                        "st1        {v0.4s}, [%0], #16  \n"
                        "subs       %w3, %w3, #1        \n"
                        "bne        0b                  \n"
                        : "=r"(destptr),
                        "=r"(nn)
                        : "0"(destptr),     // %0
                        "1"(nn),            // %w1
                        "r"(pval)           // %w4
                        : "cc", "memory", "v0"
                    );

                }

#endif

                for(; remain > 0; remain--){
                    *destptr = pval;
                    destptr++;
                }

            }

            // fill center

            for(int i = top; i < top + inHeight; i++){
                const float *srcptr = src + c * inSize +  (i - top) * inWidth;
                float *destptr = dest + c * outSize + i * outWidth;
                int remain1 = left;
                for(; remain1 > 0; remain1--){
                    *destptr = pval;
                    destptr++;
                }

                //memcpy
#if USE_ARM
                int nn = inWidth >> 2;
                int remain = inWidth - (nn << 2);
#else
                int remain = inWidth;
#endif

#if USE_NEON
                if(nn > 0){

                    asm volatile(
                        "0:                             \n"
                        "prfm       pldl1keep, [%0, #128]   \n"
                        "ld1        {v0.4s}, [%0], #16  \n"
                        "st1        {v0.4s}, [%1], #16  \n"
                        "subs       %w2, %w2, #1        \n"
                        "bne        0b                  \n"
                        : "=r"(srcptr),
                        "=r"(destptr),
                        "=r"(nn)
                        : "0"(srcptr),      // %0
                        "1"(destptr),     // %1
                        "2"(nn)           // %w2
                        : "cc", "memory", "v0"
                    );
                }
#endif

                for(; remain > 0; remain--){
                    *destptr = *srcptr;
                    srcptr++;
                    destptr++;
                }

                int remain2 = right;
                for(; remain2 > 0; remain2--){
                    *destptr = pval;
                    destptr++;
                }

            }

            //fill bottom

            for(int i = top + inHeight; i < outHeight; i++){
                float *destptr = dest + c * outSize +  i * outWidth;

#if USE_ARM
                int nn = 0;
                int remain = outWidth - (nn << 2);
#else
                int remain = outWidth;
#endif

#if USE_ARM
                if(nn > 0){
                    asm volatile(
                        "dup        v0.4s, %w4          \n"
                        "0:                             \n"
                        "st1        {v0.4s}, [%0], #16  \n"
                        "subs       %w3, %w3, #1        \n"
                        "bne        0b                  \n"
                        : "=r"(destptr),
                        "=r"(nn)
                        : "0"(destptr),     // %0
                        "1"(nn),            // %w1
                        "r"(pval)           // %w4
                        : "cc", "memory", "v0"
                    );

                }
#endif

                for(; remain > 0; remain--){
                    *destptr = pval;
                    destptr++;
                }

            }
        }
    }
}
#endif