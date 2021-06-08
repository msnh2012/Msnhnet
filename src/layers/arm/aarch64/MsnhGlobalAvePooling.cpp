#ifdef USE_ARM
#include "Msnhnet/layers/arm/aarch64/MsnhGlobalAvePooling.h"
#include "Msnhnet/config/MsnhnetCfg.h"
#include <iostream>
#include <algorithm>
using namespace std;

namespace Msnhnet
{

//bottom: src, inWidth, inHeight, inChannel
//top: dest, outWidth, outHeight, outChannel
void GlobalAvePoolingLayerArm::poolingV8(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, 
                        float* &dest, const int& ceilModel){
     const int inSize = inHeight * inWidth;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int c = 0; c < inChannel; c++){
        const float *srcptr = src + c * inSize;
        float sum = 0.f;
        float *destptr = dest + c;
#if USE_ARM
        int nn = inSize >> 2;
        int remain = inSize - (nn << 2);
#else
        int remain = inSize;
#endif

#if USE_ARM
        if(nn > 0){

            asm volatile(
                "movi       v0.2d, #0           \n"

                "0:                             \n"
                "prfm       pldl1keep, [%0, #128]   \n"
                "ld1        {v1.4s}, [%0], #16  \n"
                "fadd       v0.4s, v1.4s, v0.4s \n"

                "subs       %w1, %w1, #1        \n"
                "bne        0b                  \n"

                "faddp      v0.4s, v0.4s, v0.4s     \n"
                "faddp      v0.4s, v0.4s, v0.4s     \n"

                "fmov       %w2, s0            \n"

                : "=r"(srcptr),     // %0
                "=r"(nn),           // %1
                "=r"(sum)           // %2
                : "0"(srcptr),
                "1"(nn),
                "2"(sum)
                : "cc", "memory", "v0", "v1"
            );

        }

#endif
        for(; remain > 0; remain--){
            sum += *srcptr;
            srcptr++;
        }
        *destptr = sum / inSize;
    }
}

}
#endif