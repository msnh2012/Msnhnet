#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhAvePooling.h"
#include "Msnhnet/config/MsnhnetCfg.h"
#include <iostream>
#include <algorithm>
using namespace std;

namespace Msnhnet
{


//bottom: src, inWidth, inHeight, inChannel
//top: dest, outWidth, outHeight, outChannel
void AvePoolingLayerArm::pooling(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, 
                        float* &dest, const int& ceilModel){
     const int inSize = inHeight * inWidth;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int c = 0; c < inChannel; c++){
        const float *srcptr = src + c * inSize;
        float sum = 0.f;
        float *destptr = dest + c;
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
                "veor    q1, q1, q1             \n"
                "0:                             \n"
                "vld1.f32   {d0-d1}, [%1]!      \n"
                "vmla.f32   q1, q0, q1          \n"
                "subs       %0, #1              \n"
                "bne        0b                  \n"
                
                "vpadd.f32  d4, d2, d3          \n"
                "vpadd.f32  d5, d4, d4          \n"

                "vmov.32    %2, d5[0]           \n"

                : "=r"(nn), // %0
                "=r"(srcptr), // %1
                "=r"(sum)
                :
                : "cc", "memory", "q0", "q1", "q2", "q3"
            );
        }
#endif

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