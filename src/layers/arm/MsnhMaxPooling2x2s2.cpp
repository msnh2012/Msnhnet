#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhMaxPooling2x2s2.h"
#include "Msnhnet/config/MsnhnetCfg.h"
#include <iostream>
#include <algorithm>
using namespace std;

namespace Msnhnet
{


//bottom: src, inWidth, inHeight, inChannel
//top: dest, outWidth, outHeight, outChannel
void MaxPooling2x2s2Arm::pooling(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, 
                                    float* &dest, const int& ceilModel){
        const int outHeight = inHeight >> 1;
        const int outWidth = inWidth >> 1;
        const int inSize = inHeight * inWidth;
        const int outSize = outHeight * outWidth;
        const int tailStep = inWidth - 2 * outWidth + inWidth;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int c = 0; c < inChannel; c++){
            const float *srcptr = src + c * inSize;
            float *destptr = dest + c * outSize;
            const float* r0 = srcptr;
            const float* r1 = srcptr + inWidth;
            for(int i = 0; i < outHeight; i++){
#if USE_NEON
                int nn = outWidth >> 2;
                int remain = outWidth - (nn << 2);
#else
                int remain = outWidth;
#endif

#if USE_NEON

                if(nn > 0){
#if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "pld        [%2, #256]          \n"
                        //q0,q1 = [d0, d1, d2, d3] = [a, b, c, d]
                        "vld1.f32   {d0-d3}, [%1]!      \n"
                        //q2,d3 = [d4, d5, d6, d7] = [e, f, g, h]
                        "vld1.f32   {d4-d7}, [%2]!      \n"
                        //q0 = [max(a, e), max(b, f)]
                        "vmax.f32   q0, q0, q2          \n"
                        "vmax.f32   q1, q1, q3          \n"
                        //d4 = max([max(a, e), max(b, f)])
                        "vpmax.f32  d4, d0, d1          \n"
                        "vpmax.f32  d5, d2, d3          \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d4-d5}, [%3]!      \n"
                        "bne        0b                  \n"
                        : "=r"(nn),    // %0
                        "=r"(r0),    // %1
                        "=r"(r1),    // %2
                        "=r"(destptr) // %3
                        : "0"(nn),
                        "1"(r0),
                        "2"(r1),
                        "3"(destptr)
                        : "cc", "memory", "q0", "q1", "q2", "q3");
#endif
                }

#endif

                for(; remain > 0; remain--){
                    float mx0 = std::max(r0[0], r0[1]);
                    float mx1 = std::max(r1[0], r1[1]);
                    *destptr = std::max(mx0, mx1);
                    r0 += 2;
                    r1 += 2;
                    destptr++;
                }

                r0 += tailStep;
                r1 += tailStep;
            }

        }
    }


}
#endif