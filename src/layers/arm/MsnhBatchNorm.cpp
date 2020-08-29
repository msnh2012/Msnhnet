#include "Msnhnet/layers/arm/MsnhBatchNorm.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    void BatchNormLayerArm::BatchNorm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float* &dest,
                    float *const &Scales, float *const &rollMean, float *const &rollVariance, float *const &biases){
        // a = bias - slope * mean / sqrt(var)
        // b = slope / sqrt(var)
        // value = b * value + a
        int in_size = inWidth * inHeight;
        const float *srcPtr = src;
        float *destPtr = dest;
        int nn, remain;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif

        for(int i = 0; i < inChannel; i++){

            float sqrtVar = sqrt(rollVariance[i] + 0.00001f);
            float a = biases[i] - Scales[i] * rollMean[i] / sqrtVar;
            float b = Scales[i] / sqrtVar;

            #if USE_NEON
                nn = in_size >> 2;
                remain = in_size - nn << 2;
                float32x4_t a_new = vdupq_n_f32(a);
                float32x4_t b_new = vdupq_n_f32(b);
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
                    #if __aarch64__
                    #else
                        asm volatile(
                            "vdup.f32   q0, %6              \n"
                            "vdup.f32   q1, %7              \n"

                            "0:                             \n"
                            "pld        [%2, #128]          \n"
                            "vld1.f32   {d4,d5}, [%1]!      \n"

                            "vmul.f32   q3, q2, q1          \n"
                            "vadd.f32   q3, q3, q0          \n"

                            "vst1.f32   {d6-d7}, [%2]!    \n"

                            : "=r"(nn),     // %0
                            "=r"(srcPtr), // %1
                            "=r"(destPtr),     // %2
                            : "=r"(nn),     // %0
                            "=r"(srcPtr), // %4
                            "=r"(destPtr),     // %5
                            "w"(a_new), // %6
                            "w"(b_new), // %7
                            : "cc", "memory", "q0", "q1", "q2", "q3");
                        );
                    #endif
                }

            #endif

            for(; remain > 0; remain--){
                *destPtr = b * (*srcPtr) + a;
                srcPtr++;
                destPtr++;
            }
        }
    }

}