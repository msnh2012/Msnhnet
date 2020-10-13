#define USE_ARM 1
#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolutionDepthwise3x3s1.h"
namespace Msnhnet
{
    void ConvolutionalDepthwiseLayerArm3x3s1::convdepthwise3x3s1Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        int ccOutChannel = outChannel >> 1;
        int ccRemainOutChannel = ccOutChannel << 1;

        const int in_size = inWidth * inHeight;
        const int out_size = outWidth * outHeight;
        const int group = inChannel;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif 
        for(int g = 0; g < group; g++){
            float *dest0 = dest + g * out_size;
            const float* kernel0 = kernel + g * 9;

            float* destptr0 = dest0;
            float* destptr0_next = destptr0 + outWidth;

            const float* src0 = src + g * in_size;

            const float* r0 = src0;
            const float* r1 = src0 + inWidth;
            const float* r2 = src0 + inWidth * 2;
            const float* r3 = src0 + inWidth * 3;

#if USE_NEON
            float32x4_t k012 = vld1q_f32(k0);
            float32x4_t k345 = vld1q_f32(k0 + 3);
            float32x4_t k678 = vld1q_f32(k0 + 6);
            float32x4_t bias0 = vdupq_n_f32(0.f);
#else
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;
#endif 
            int i = 0;

            for(; i + 1 < outHeight; i += 2){
#if USE_NEON
                int nn = outWidth >> 2;
                int remain = outWidth & 3;
#else
                int remain = outWidth;
#endif

#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                if(nn > 0){
                    
                }
#endif
#endif
                for(; remain > 0; remain--){
#if USE_NEON
                    float32x4_t r00 = vld1q_f32(r0);
                    float32x4_t r10 = vld1q_f32(r1);
                    float32x4_t r20 = vld1q_f32(r2);
                    float32x4_t r30 = vld1q_f32(r3);

                    float32x4_t sum1 = vmulq_f32(r00, k012);
                    sum1 = vmlaq_f32(sum1, r10, k345);
                    sum1 = vmlaq_f32(sum1, r20, k678);

                    float32x4_t sum2 = vmulq_f32(r10, k012);
                    sum2 = vmlaq_f32(sum2, r20, k345);
                    sum2 = vmlaq_f32(sum2, r30, k678);

                    sum1 = vsetq_lane_f32(bias0, sum1, 3);
                    sum2 = vsetq_lane_f32(bias0, sum2, 3);

#if __aarch64__
                    *destptr0 = vaddvq_f32(sum1);
                    *destptr0_next = vaddvq_f32(sum2);
#else
                    float32x2_t a = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    float32x2_t b = vadd_f32(vget_low_f32(sum2), vget_high_f32(sum2));

                    float32x2_t c = vpadd_f32(a, b);

                    *destptr0 = vget_lane_f32(c, 0);
                    *destptr0_next = vget_lane_f32(c, 1);
#endif

#else
                    float sum1 = 0.f;
                    sum1 += r0[0] * k0[0];
                    sum1 += r0[1] * k0[1];
                    sum1 += r0[2] * k0[2];
                    sum1 += r1[0] * k1[0];
                    sum1 += r1[1] * k1[1];
                    sum1 += r1[2] * k1[2];
                    sum1 += r2[0] * k2[0];
                    sum1 += r2[1] * k2[1];
                    sum1 += r2[2] * k2[2];

                    float sum2 = 0.f;
                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *destptr0 = sum1;
                    *destptr0_next = sum2; 
#endif
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    destptr0++;
                    destptr0_next++;
                }

                r0 += 2 + inWidth;
                r1 += 2 + inWidth;
                r2 += 2 + inWidth;
                r3 += 2 + inWidth;

                destptr0 += outWidth;
                destptr0_next += outWidth;
            }


        }
    }
}

#endif