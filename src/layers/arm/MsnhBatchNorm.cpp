#include "Msnhnet/layers/arm/MsnhBatchNorm.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    void BatchNormLayerArm::BatchNorm(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel,
                    float *const &Scales, float *const &rollMean, float *const &rollVariance, float *const &biases){
        // a = bias - slope * mean / sqrt(var)
        // b = slope / sqrt(var)
        // value = b * value + a
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif

        int in_size = inWidth * inHeight;
        float *srcPtr = src;
        int nn, remain;
        cout << inChannel << " " << in_size << endl;
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
                for(; nn > 0; nn--){
                    #if __aarch64__
                        throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
                    #else
                        float32x4_t tmp = vld1q_f32(srcPtr);
                        float32x4_t sum = vmulq_f32(tmp, b_new);
                        sum = vaddq_f32(sum, a_new);
                        vst1q_f32(srcPtr, sum);
                        srcPtr += 4;
                    #endif 
                }
            #endif

            for(; remain > 0; remain--){
                *srcPtr = b * (*srcPtr) + a;
                srcPtr++;
            }
        }

    }

}