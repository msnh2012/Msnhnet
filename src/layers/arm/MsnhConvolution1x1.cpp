#define USE_ARM 1
#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolution1x1.h"

namespace Msnhnet
{
    void ConvolutionalLayerArm1x1::conv1x1s1Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        int ccOutChannel = outChannel >> 2;
        int ccRemainOutChannel = ccOutChannel << 2;

        const int in_size = inWidth * inHeight;
        const int out_size = outWidth * outHeight;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif 
        for(int cc = 0; cc < ccOutChannel; cc++){
            int c = cc << 2;
            
            float *dest0 = dest + c * out_size;
            float *dest1 = dest + (c + 1) * out_size;
            float *dest2 = dest + (c + 2) * out_size;
            float *dest3 = dest + (c + 3) * out_size;

            int q = 0;

            for(q = 0; q + 3 < inChannel; q += 4){
                float *destptr0 = dest0;
                float *destptr1 = dest1;
                float *destptr2 = dest2;
                float *destptr3 = dest3;

                const float *src0 = src + q * in_size;
                const float *src1 = src + (q + 1) * in_size;
                const float *src2 = src + (q + 2) * in_size;
                const float *src3 = src + (q + 3) * in_size;

                const float *r0 = src0;
                const float *r1 = src1;
                const float *r2 = src2;
                const float *r3 = src3;

                const float *kernel0 = kernel + c * inChannel + q;
                const float *kernel1 = kernel + (c + 1) * inChannel + q;
                const float *kernel2 = kernel + (c + 2) * inChannel + q;
                const float *kernel3 = kernel + (c + 3) * inChannel + q;

#if USE_NEON
                int nn = out_size >> 3;
                int remain = out_size & 7;
                float32x4_t k0 = vld1q_f32(kernel0);
                float32x4_t k1 = vld1q_f32(kernel1);
                float32x4_t k2 = vld1q_f32(kernel2);
                float32x4_t k3 = vld1q_f32(kernel3);
#else
                int remain = out_size;
#endif
                
#if USE_NEON
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                if(nn > 0){
                    asm volatile(
                        
                    

                        : "=r"(nn),      // %0
                        "=r"(destptr0), // %1
                        "=r"(destptr1), // %2
                        "=r"(destptr2), // %3
                        "=r"(destptr3), // %4
                        "=r"(r0),      // %5
                        "=r"(r1),      // %6
                        "=r"(r2),      // %7
                        "=r"(r3)       // %8
                        : "0"(nn),
                        "1"(destptr0),
                        "2"(destptr1),
                        "3"(destptr2),
                        "4"(destptr3),
                        "5"(r0),
                        "6"(r1),
                        "7"(r2),
                        "8"(r3),
                        "w"(k0), // %18
                        "w"(k1), // %19
                        "w"(k2), // %20
                        "w"(k3)  // %21
                        : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                    );
                }
#endif
#endif

                for(; remain > 0;  remain--){
                    float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                    float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                    float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                    float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];

                    *destptr0 += sum0;
                    *destptr1 += sum1;
                    *destptr2 += sum2;
                    *destptr3 += sum3;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    destptr0++;
                    destptr1++;
                    destptr2++;
                    destptr3++;
                }
            }

            for(; q < inChannel; q++){
                float *destptr0 = dest0;
                float *destptr1 = dest1;
                float *destptr2 = dest2;
                float *destptr3 = dest3;

                const float *src0 = src + q * in_size;

                const float *kernel0 = kernel + c * inChannel + q;
                const float *kernel1 = kernel + (c + 1) * inChannel + q;
                const float *kernel2 = kernel + (c + 2) * inChannel + q;
                const float *kernel3 = kernel + (c + 3) * inChannel + q;

                const float *r0 = src0;

#if USE_NEON
                int nn = out_size >> 3;
                int remain = out_size & 7;
                float32x4_t k0 = vld1q_f32(kernel0);
                float32x4_t k1 = vld1q_f32(kernel1);
                float32x4_t k2 = vld1q_f32(kernel2);
                float32x4_t k3 = vld1q_f32(kernel3);
#else
                int remain = out_size;
#endif

#if USE_NEON
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                if(nn > 0){
                    asm volatile(

                    );
                }
#endif
#endif
                for(; remain > 0; remain--){
                    float sum0 = *r0 * kernel0[0];
                    float sum1 = *r0 * kernel1[0];
                    float sum2 = *r0 * kernel2[0];
                    float sum3 = *r0 * kernel3[0];

                    *destptr0 += sum0;
                    *destptr1 += sum1;
                    *destptr2 += sum2;
                    *destptr3 += sum3;

                    r0++;
                    destptr0++;
                    destptr1++;
                    destptr2++;
                    destptr3++;
                }

            }
        }

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif 
        for(int cc = ccRemainOutChannel; cc < outChannel; cc++){
            float *dest0 = dest + cc * out_size;

            int q = 0;

            for(; q + 3 < inChannel; q += 4){
                float *destptr0 = dest0;

                const float *src0 = src + q * in_size;
                const float *src1 = src + (q + 1) * in_size;
                const float *src2 = src + (q + 2) * in_size;
                const float *src3 = src + (q + 3) * in_size;

                const float *r0 = src0;
                const float *r1 = src1;
                const float *r2 = src2;
                const float *r3 = src3;

                const float *kernel0 = kernel + cc * inChannel + q;

#if USE_NEON
                int nn = out_size >> 3;
                int remain = out_size & 7;
                float32x4_t k0 = vdupq_n_f32(kernel0[0]);
                float32x4_t k1 = vdupq_n_f32(kernel0[1]);
                float32x4_t k2 = vdupq_n_f32(kernel0[2]);
                float32x4_t k3 = vdupq_n_f32(kernel0[3]);
#else
                int remain = out_size;
#endif
                
            }
        }
    }    

    void ConvolutionalLayerArm1x1::conv1x1s2Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        
    }    
}

#endif