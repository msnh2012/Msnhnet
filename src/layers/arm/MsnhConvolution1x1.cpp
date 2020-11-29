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
                        "0:                                 \n"

                        // float *destptr0 = dest0;
                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1]         \n"

                        // float *destptr1 = dest1;
                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2]         \n"
                        // float *destptr2 = dest2;
                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3]         \n"

                        // float *destptr3 = dest3;
                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4]         \n"

                        // const float *r0 = src0;
                        "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5]!        \n"

                        // float sum0 = *r0 * kernel0[0]
                        "vmla.f32   q8, q6, %e18[0]         \n"
                        // float sum0_next = *r0 * kernel0[0]
                        "vmla.f32   q9, q7, %e18[0]         \n"

                        // float sum1 = *r0 * kernel1[0]
                        "vmla.f32   q10, q6, %e19[0]        \n"
                        // float sum1_next = *r0 * kernel1[0]
                        "vmla.f32   q11, q7, %e19[0]        \n"

                        // float sum2 = *r0 * kernel2[0]
                        "vmla.f32   q12, q6, %e20[0]        \n"
                        // float sum2_next = *r0 * kernel2[0]
                        "vmla.f32   q13, q7, %e20[0]        \n"

                        // float sum3 = *r0 * kernel3[0]
                        "vmla.f32   q14, q6, %e21[0]        \n"
                        // float sum3_next = *r0 * kernel3[0]
                        "vmla.f32   q15, q7, %e21[0]        \n"
                        
                        // const float *r1 = src1;
                        "pld        [%6, #256]              \n"
                        "vld1.f32   {d8-d11}, [%6]!         \n"


                        // float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1]
                        "vmla.f32   q8, q4, %e18[1]         \n"
                        // float sum0_next = *r0 * kernel0[0] + *r1 * kernel0[1]
                        "vmla.f32   q9, q5, %e18[1]         \n"

                        // float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1]
                        "vmla.f32   q10, q4, %e19[1]        \n"
                        // float sum1_next = *r0 * kernel1[0] + *r1 * kernel1[1]
                        "vmla.f32   q11, q5, %e19[1]        \n"

                        // float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1]
                        "vmla.f32   q12, q4, %e20[1]        \n"
                        // float sum2_next = *r0 * kernel2[0] + *r1 * kernel2[1]
                        "vmla.f32   q13, q5, %e20[1]        \n"

                        // float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1]
                        "vmla.f32   q14, q4, %e21[1]        \n"
                        // float sum3_next = *r0 * kernel3[0] + *r1 * kernel3[1]
                        "vmla.f32   q15, q5, %e21[1]        \n"

                        // const float *r2 = src2;
                        "pld        [%7, #256]              \n"
                        "vld1.f32   {d12-d15}, [%7]!        \n"

                        // float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2]
                        "vmla.f32   q8, q6, %f18[0]         \n"
                        // float sum0_next = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2]
                        "vmla.f32   q9, q7, %f18[0]         \n"

                        //  float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2]
                        "vmla.f32   q10, q6, %f19[0]        \n"
                        //  float sum1_next = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2]
                        "vmla.f32   q11, q7, %f19[0]        \n"

                        // float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2]
                        "vmla.f32   q12, q6, %f20[0]        \n"
                        // float sum2_next = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2]
                        "vmla.f32   q13, q7, %f20[0]        \n"

                        // float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2]
                        "vmla.f32   q14, q6, %f21[0]        \n"
                        // float sum3_next = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2]
                        "vmla.f32   q15, q7, %f21[0]        \n"

                        // const float *r3 = src3;
                        "pld        [%8, #256]              \n"
                        "vld1.f32   {d8-d11}, [%8]!         \n"

                        // float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                        "vmla.f32   q8, q4, %f18[1]         \n"
                        // float sum0_next = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                        "vmla.f32   q9, q5, %f18[1]         \n"

                        // float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                        "vmla.f32   q10, q4, %f19[1]        \n"
                        // float sum1_next = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                        "vmla.f32   q11, q5, %f19[1]        \n"

                        // float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                        "vmla.f32   q12, q4, %f20[1]        \n"
                        // float sum2_next = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                        "vmla.f32   q13, q5, %f20[1]        \n"

                        // float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];
                        "vmla.f32   q14, q4, %f21[1]        \n"
                        // float sum3_next = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];
                        "vmla.f32   q15, q5, %f21[1]        \n"

                        // *destptr0 += sum0;
                        "vst1.f32   {d16-d19}, [%1]!        \n"
                        // *destptr1 += sum1;
                        "vst1.f32   {d20-d23}, [%2]!        \n"
                        // *destptr2 += sum2;
                        "vst1.f32   {d24-d27}, [%3]!        \n"
                        // *destptr3 += sum3;
                        "vst1.f32   {d28-d31}, [%4]!        \n"

                        // nn-=1
                        "subs       %0, #1                  \n"
                        "bne        0b                      \n"

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
                        "0:                                 \n"

                        // float *destptr0 = dest0;
                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1]         \n"

                        // float *destptr1 = dest1;
                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2]         \n"
                        // float *destptr2 = dest2;
                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3]         \n"

                        // float *destptr3 = dest3;
                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4]         \n"

                        // const float *r0 = src0;
                        "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5]!        \n"

                        // float sum0 = *r0 * kernel0[0]
                        "vmla.f32   q8, q6, %e18[0]         \n"
                        // float sum0_next = *r0 * kernel0[0]
                        "vmla.f32   q9, q7, %e18[0]         \n"

                        // float sum1 = *r0 * kernel1[0]
                        "vmla.f32   q10, q6, %e19[0]        \n"
                        // float sum1_next = *r0 * kernel1[0]
                        "vmla.f32   q11, q7, %e19[0]        \n"

                        // float sum2 = *r0 * kernel2[0]
                        "vmla.f32   q12, q6, %e20[0]        \n"
                        // float sum2_next = *r0 * kernel2[0]
                        "vmla.f32   q13, q7, %e20[0]        \n"

                        // float sum3 = *r0 * kernel3[0]
                        "vmla.f32   q14, q6, %e21[0]        \n"
                        // float sum3_next = *r0 * kernel3[0]
                        "vmla.f32   q15, q7, %e21[0]        \n"
                        
                        // *destptr0 += sum0;
                        "vst1.f32   {d16-d19}, [%1]!        \n"
                        // *destptr1 += sum1;
                        "vst1.f32   {d20-d23}, [%2]!        \n"
                        // *destptr2 += sum2;
                        "vst1.f32   {d24-d27}, [%3]!        \n"
                        // *destptr3 += sum3;
                        "vst1.f32   {d28-d31}, [%4]!        \n"

                        // nn-=1
                        "subs       %0, #1                  \n"
                        "bne        0b                      \n"

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
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
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

#if USE_NEON
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                if(nn > 0){
                    asm volatile(
                        "0:                             \n"
                        // const float *r0 = src0;
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2]!      \n"
                        // float *destptr0 = dest0;
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1]       \n"

                        // float sum0 = *r0 * kernel0[0]
                        "vmla.f32   q0, q2, %q12        \n"
                        // float sum0_next = *r0 * kernel0[0]
                        "vmla.f32   q1, q3, %q12        \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d4-d7}, [%3]!      \n"
                        "vmla.f32   q0, q2, %q13        \n"

                        "vmla.f32   q1, q3, %q13        \n"
                        
                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d4-d7}, [%4]!      \n"
                        
                        "vmla.f32   q0, q2, %q14        \n"
                        "vmla.f32   q1, q3, %q14        \n"
                        
                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5]!      \n"
                        
                        "vmla.f32   q0, q2, %q15        \n"
                        "vmla.f32   q1, q3, %q15        \n"

                        "vst1.f32   {d0-d3}, [%1]!      \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        : "=r"(nn),     // %0
                        "=r"(dest0), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(r3)      // %5
                        : "0"(nn),
                        "1"(dest0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "w"(k0), // %12
                        "w"(k1), // %13
                        "w"(k2), // %14
                        "w"(k3)  // %15
                        : "cc", "memory", "q0", "q1", "q2", "q3"
                    );
                }
#endif
#endif

                for(; remain > 0; remain--){
                    float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];

                    *destptr0 += sum0;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    destptr0++;
                }

            }

            for(; q < inChannel; q++){

                float *destptr0 = dest0;
                const float *src0 = src + q * in_size;
                const float *kernel0 = kernel + cc * inChannel + q;
                const float *r0 = src0;
#if USE_NEON
                int nn = out_size >> 3;
                int remain = out_size & 7;
                float32x4_t k0 = vdupq_n_f32(kernel0[0]);
#else
                int remain = out_size;
#endif

#if USE_NEON
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                if(nn > 0){
                    asm volatile(
                    "0:                             \n"
                    "pld        [%2, #256]          \n"
                    "vld1.f32   {d4-d7}, [%2]!      \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]       \n"
                    
                    "vmla.f32   q0, q2, %q6         \n"
                    "vmla.f32   q1, q3, %q6         \n"

                    "vst1.f32   {d0-d3}, [%1]!      \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"

                    : "=r"(nn),     // %0
                    "=r"(destptr0), // %1
                    "=r"(r0)      // %2
                    : "0"(nn),
                    "1"(destptr0),
                    "2"(r0),
                    "w"(k0) // %6
                    : "cc", "memory", "q0", "q1", "q2", "q3"
                    );
                }
#endif
#endif
                for(; remain > 0; remain--){
                    float sum0 = *r0 * kernel0[0];

                    *destptr0 += sum0;

                    r0++;
                    destptr0++;
                }

            }
        }
    }    

    void ConvolutionalLayerArm1x1::conv1x1s2Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        
    }    

    // pack 4x4
    // shape[c, h, w]: [outChannel / 4 + outChannel %4， 4 * 4， inChannel / 4 + inChannel%4]
    void ConvolutionalLayerArm1x1::conv1x1s1SgemmTransformKenel(float *const &kernel, float* &dest, const int &inChannel, const int &outChannel){
        int c = 0;

        int Stride = 4 * 4 * (inChannel / 4 + inChannel%4);

        for(; c + 3 < outChannel; c += 4){
            const float* k0 = kernel + c * inChannel;
            const float* k1 = kernel + (c + 1) * inChannel;
            const float* k2 = kernel + (c + 2) * inChannel;
            const float* k3 = kernel + (c + 3) * inChannel;

            float* destptr = dest + (c / 4) * Stride;

            for(int i = 0; i < inChannel; i++){

                destptr[0] = k0[0];
                destptr[1] = k1[0];
                destptr[2] = k2[0];
                destptr[3] = k3[0];

                destptr += 4;

                k0 += 1;
                k1 += 1;
                k2 += 1;
                k3 += 1;
            }
        }

        for(; c < outChannel; c++){
            const float* k0 = kernel + c * inChannel;

            float* destptr = dest + (c / 4 + c % 4) * Stride;

            for(int i = 0; i < inChannel; i++){
                destptr[0] = k0[0];
                destptr += 4;
                k0 += 1;
            }
        }
    }

    // pack 8x4
    // shape[c, h, w]: [outSize / 8 + (outSize % 8) / 4 + outSize % 4, 8*4, inChannel/4+inChannel%4]
    void ConvolutionalLayerArm1x1::conv1x1s1SgemmNeon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        int outSize = outHeight * outWidth;
        // transformed kernel
        int kernelSize = 4 * 4 * (inChannel / 4 + inChannel%4);

        // pack input start
        int nnSize = outSize >> 3;
        int remainSize = nnSize << 3;

        int src_tm_channel = outSize / 8 + (outSize % 8) / 4 + outSize % 4;
        int src_tm_h = 8 * 4;
        int src_tm_w = inChannel/4+inChannel%4;
        int src_tm_size = src_tm_h * src_tm_w;
        float *src_tm = new float[src_tm_channel * src_tm_size];

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i = 0; i < nnSize; i++){
            int newi = i << 3;
            const float* srcptr = src + newi;

            float *src_tm_ptr = src_tm + (newi / 8) * src_tm_size;

            for(int q = 0; q < inChannel; q++){
#if USE_NEON
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0]       \n"
                    "vst1.f32   {d0-d3}, [%1]!      \n"
                    : "=r"(srcptr),  // %0
                    "=r"(src_tm_ptr) // %1
                    : "0"(srcptr),
                    "1"(src_tm_ptr)
                    : "memory", "q0", "q1"
                );
#else
                src_tm_ptr[0] = srcptr[0];
                src_tm_ptr[1] = srcptr[1];
                src_tm_ptr[2] = srcptr[2];
                src_tm_ptr[3] = srcptr[3];
                src_tm_ptr[4] = srcptr[4];
                src_tm_ptr[5] = srcptr[5];
                src_tm_ptr[6] = srcptr[6];
                src_tm_ptr[7] = srcptr[7];
                src_tm_ptr += 8;
#endif
                srcptr += outSize;
            }

        }

        nnSize = (outSize - remainSize) >> 2;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i = 0; i < nnSize; i++){
            int newi = remainSize + i * 4;

            const float* srcptr = src + newi;

            float *src_tm_ptr = src_tm + (newi / 8 + (newi % 8) / 4) * src_tm_size;

            for(int q = 0; q < inChannel; q++){
#if USE_NEON
                asm volatile(
                    "pld        [%0, #128]          \n"
                    "vld1.f32   {d0-d1}, [%0]       \n"
                    "vst1.f32   {d0-d1}, [%1]!      \n"
                    : "=r"(srcptr),  // %0
                    "=r"(src_tm_ptr) // %1
                    : "0"(srcptr),
                    "1"(src_tm_ptr)
                    : "memory", "q0"
                );
#else
                src_tm_ptr[0] = srcptr[0];
                src_tm_ptr[1] = srcptr[1];
                src_tm_ptr[2] = srcptr[2];
                src_tm_ptr[3] = srcptr[3];
                src_tm_ptr += 4;

#endif
                srcptr += outSize;
            }
        }

        remainSize += nnSize << 2;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif

        for(int i = remainSize; i < outSize; i++){
            int newi = i;

            const float* srcptr = src + newi;

            float *src_tm_ptr = src_tm + (newi / 8 + (newi % 8) / 4 + newi % 4) * src_tm_size;

            for(int q = 0; q < inChannel; q++){

                src_tm_ptr[0] = srcptr[0];

                src_tm_ptr += 1;
                srcptr += inHeight * inWidth;
            }
        }

        // pack input end

        int nnOutChannel = outChannel >> 2;
        int remainOutChannel = nnOutChannel << 2;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = 0; cc < nnOutChannel; cc++){
            int c = cc << 2;

            float *destptr0 = dest + c * outSize;
            float *destptr1 = dest + (c + 1) * outSize;
            float *destptr2 = dest + (c + 2) * outSize;
            float *destptr3 = dest + (c + 3) * outSize;

            int i = 0;

            for(; i + 7 < outSize; i += 8){
                const float *src_tm_ptr = src_tm + (i / 8) * src_tm_size;

                const float *kernel0 = kernel + (c / 4) *  kernelSize;

#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(

                    "veor       q0, q0, q0          \n"
                    "vdup.f32   q8, d0[0]           \n"
                    "vdup.f32   q9, d0[0]           \n"
                    "vdup.f32   q10, d0[0]          \n"
                    "vdup.f32   q11, d0[0]          \n"

                    "vdup.f32   q12, d0[0]          \n"
                    "vdup.f32   q13, d0[0]          \n"
                    "vdup.f32   q14, d0[0]          \n"
                    "vdup.f32   q15, d0[0]          \n"
                    // r4 = inChannnel >> 2
                    "lsr        r4, %12, #2         \n"
                    "cmp        r4, #0              \n"
                    "beq        1f                  \n"

                    "0:                             \n"
                    "pld        [%4, #512]          \n"
                    "vldm       %4!, {d8-d15}       \n"

                    "pld        [%5, #512]          \n"
                    "vldm       %5!, {d0-d7}        \n"

                    "vmla.f32   q8, q4, d0[0]       \n"
                    "vmla.f32   q10, q4, d0[1]      \n"
                    "vmla.f32   q12, q4, d1[0]      \n"
                    "vmla.f32   q14, q4, d1[1]      \n"

                    "vmla.f32   q9, q5, d0[0]       \n"
                    "vmla.f32   q11, q5, d0[1]      \n"
                    "vmla.f32   q13, q5, d1[0]      \n"
                    "vmla.f32   q15, q5, d1[1]      \n"

                    "vmla.f32   q8, q6, d2[0]       \n"
                    "vmla.f32   q10, q6, d2[1]      \n"
                    "vmla.f32   q12, q6, d3[0]      \n"
                    "vmla.f32   q14, q6, d3[1]      \n"

                    "vmla.f32   q9, q7, d2[0]       \n"
                    "vmla.f32   q11, q7, d2[1]      \n"
                    "vmla.f32   q13, q7, d3[0]      \n"
                    "vmla.f32   q15, q7, d3[1]      \n"

                    "pld        [%4, #512]          \n"
                    "vldm       %4!, {d8-d15}       \n"

                    "vmla.f32   q8, q4, d4[0]       \n"
                    "vmla.f32   q10, q4, d4[1]      \n"
                    "vmla.f32   q12, q4, d5[0]      \n"
                    "vmla.f32   q14, q4, d5[1]      \n"

                    "vmla.f32   q9, q5, d4[0]       \n"
                    "vmla.f32   q11, q5, d4[1]      \n"
                    "vmla.f32   q13, q5, d5[0]      \n"
                    "vmla.f32   q15, q5, d5[1]      \n"

                    "vmla.f32   q8, q6, d6[0]       \n"
                    "vmla.f32   q10, q6, d6[1]      \n"
                    "vmla.f32   q12, q6, d7[0]      \n"
                    "vmla.f32   q14, q6, d7[1]      \n"

                    "vmla.f32   q9, q7, d6[0]       \n"
                    "vmla.f32   q11, q7, d6[1]      \n"
                    "vmla.f32   q13, q7, d7[0]      \n"
                    "vmla.f32   q15, q7, d7[1]      \n"

                    "subs       r4, r4, #1          \n"
                    "bne        0b                  \n"

                    "1:                             \n"
                    // r4 = remain = inChannel & 3;
                    "and        r4, %12, #3         \n" 
                    "cmp        r4, #0              \n"
                    "beq        3f                  \n"

                    "2:                             \n"

                    "pld        [%4, #256]          \n"
                    "vld1.f32   {d8-d11}, [%4]!     \n"

                    "pld        [%5, #128]          \n"
                    "vld1.f32   {d0-d1}, [%5]!      \n"

                    "vmla.f32   q8, q4, d0[0]       \n"
                    "vmla.f32   q10, q4, d0[1]      \n"
                    "vmla.f32   q12, q4, d1[0]      \n"
                    "vmla.f32   q14, q4, d1[1]      \n"

                    "vmla.f32   q9, q5, d0[0]       \n"
                    "vmla.f32   q11, q5, d0[1]      \n"
                    "vmla.f32   q13, q5, d1[0]      \n"
                    "vmla.f32   q15, q5, d1[1]      \n"

                    
                    "subs       r4, r4, #1          \n"

                    "bne        2b                  \n"

                    "3:                             \n"

                    "vst1.f32   {d16-d19}, [%0]!   \n"
                    "vst1.f32   {d20-d23}, [%1]!   \n"
                    "vst1.f32   {d24-d27}, [%2]!   \n"
                    "vst1.f32   {d28-d31}, [%3]!   \n"
                    

                    : "=r"(destptr0), // %0
                    "=r"(destptr1), // %1
                    "=r"(destptr2), // %2
                    "=r"(destptr3), // %3
                    "=r"(src_tm_ptr),  // %4
                    "=r"(kernel0)     // %5
                    : "0"(destptr0),
                    "1"(destptr1),
                    "2"(destptr2),
                    "3"(destptr3),
                    "4"(src_tm_ptr),
                    "5"(kernel0),
                    "r"(inChannel)     // %12
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif

#else
                float sum0_0 = 0.f;
                float sum0_1 = 0.f;
                float sum0_2 = 0.f;
                float sum0_3 = 0.f;
                float sum0_4 = 0.f;
                float sum0_5 = 0.f;
                float sum0_6 = 0.f;
                float sum0_7 = 0.f;

                float sum1_0 = 0.f;
                float sum1_1 = 0.f;
                float sum1_2 = 0.f;
                float sum1_3 = 0.f;
                float sum1_4 = 0.f;
                float sum1_5 = 0.f;
                float sum1_6 = 0.f;
                float sum1_7 = 0.f;

                float sum2_0 = 0.f;
                float sum2_1 = 0.f;
                float sum2_2 = 0.f;
                float sum2_3 = 0.f;
                float sum2_4 = 0.f;
                float sum2_5 = 0.f;
                float sum2_6 = 0.f;
                float sum2_7 = 0.f;

                float sum3_0 = 0.f;
                float sum3_1 = 0.f;
                float sum3_2 = 0.f;
                float sum3_3 = 0.f;
                float sum3_4 = 0.f;
                float sum3_5 = 0.f;
                float sum3_6 = 0.f;
                float sum3_7 = 0.f;

                for(int q = 0; q < inChannel; q++){
                    sum0_0 += src_tm_ptr[0] * kernel0[0];
                    sum0_1 += src_tm_ptr[1] * kernel0[0];
                    sum0_2 += src_tm_ptr[2] * kernel0[0];
                    sum0_3 += src_tm_ptr[3] * kernel0[0];
                    sum0_4 += src_tm_ptr[4] * kernel0[0];
                    sum0_5 += src_tm_ptr[5] * kernel0[0];
                    sum0_6 += src_tm_ptr[6] * kernel0[0];
                    sum0_7 += src_tm_ptr[7] * kernel0[0];

                    sum1_0 += src_tm_ptr[0] * kernel0[1];
                    sum1_1 += src_tm_ptr[1] * kernel0[1];
                    sum1_2 += src_tm_ptr[2] * kernel0[1];
                    sum1_3 += src_tm_ptr[3] * kernel0[1];
                    sum1_4 += src_tm_ptr[4] * kernel0[1];
                    sum1_5 += src_tm_ptr[5] * kernel0[1];
                    sum1_6 += src_tm_ptr[6] * kernel0[1];
                    sum1_7 += src_tm_ptr[7] * kernel0[1];

                    sum2_0 += src_tm_ptr[0] * kernel0[2];
                    sum2_1 += src_tm_ptr[1] * kernel0[2];
                    sum2_2 += src_tm_ptr[2] * kernel0[2];
                    sum2_3 += src_tm_ptr[3] * kernel0[2];
                    sum2_4 += src_tm_ptr[4] * kernel0[2];
                    sum2_5 += src_tm_ptr[5] * kernel0[2];
                    sum2_6 += src_tm_ptr[6] * kernel0[2];
                    sum2_7 += src_tm_ptr[7] * kernel0[2];

                    sum3_0 += src_tm_ptr[0] * kernel0[3];
                    sum3_1 += src_tm_ptr[1] * kernel0[3];
                    sum3_2 += src_tm_ptr[2] * kernel0[3];
                    sum3_3 += src_tm_ptr[3] * kernel0[3];
                    sum3_4 += src_tm_ptr[4] * kernel0[3];
                    sum3_5 += src_tm_ptr[5] * kernel0[3];
                    sum3_6 += src_tm_ptr[6] * kernel0[3];
                    sum3_7 += src_tm_ptr[7] * kernel0[3];

                    src_tm_ptr += 8;
                    kernel0 += 4;
                }    

                destptr0[0] = sum0_0;
                destptr0[1] = sum0_1;
                destptr0[2] = sum0_2;
                destptr0[3] = sum0_3;
                destptr0[4] = sum0_4;
                destptr0[5] = sum0_5;
                destptr0[6] = sum0_6;
                destptr0[7] = sum0_7;

                destptr1[0] = sum1_0;
                destptr1[1] = sum1_1;
                destptr1[2] = sum1_2;
                destptr1[3] = sum1_3;
                destptr1[4] = sum1_4;
                destptr1[5] = sum1_5;
                destptr1[6] = sum1_6;
                destptr1[7] = sum1_7;

                destptr2[0] = sum2_0;
                destptr2[1] = sum2_1;
                destptr2[2] = sum2_2;
                destptr2[3] = sum2_3;
                destptr2[4] = sum2_4;
                destptr2[5] = sum2_5;
                destptr2[6] = sum2_6;
                destptr2[7] = sum2_7;

                destptr3[0] = sum3_0;
                destptr3[1] = sum3_1;
                destptr3[2] = sum3_2;
                destptr3[3] = sum3_3;
                destptr3[4] = sum3_4;
                destptr3[5] = sum3_5;
                destptr3[6] = sum3_6;
                destptr3[7] = sum3_7;

                destptr0 += 8;
                destptr1 += 8;
                destptr2 += 8;
                destptr3 += 8;
#endif

            }

            for(; i + 3 < outSize; i += 4){
                const float *src_tm_ptr = src_tm + ((i / 8) + (i % 8) / 4) * src_tm_size;
                const float *kernel0 = kernel + (c / 4) *  kernelSize;

#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    "veor       q0, q0, q0          \n"
                    "vdup.f32   q8, d0[0]           \n"
                    "vdup.f32   q9, d0[0]           \n"
                    "vdup.f32   q10, d0[0]          \n"
                    "vdup.f32   q11, d0[0]          \n"

                    // r4 = nn = inChannel >> 2
                    "lsr        r4, %12, #2         \n" 
                    "cmp        r4, #0              \n"
                    "beq        1f                  \n"

                    "0:                             \n"
                    "pld        [%4, #512]          \n"
                    "vldm       %4!, {d8-d15}       \n"

                    "pld        [%5, #512]          \n"
                    "vldm       %5!, {d0-d7}        \n"

                    "vmla.f32   q8, q4, d0[0]       \n"
                    "vmla.f32   q9, q4, d0[1]       \n"
                    "vmla.f32   q10, q4, d1[0]      \n"
                    "vmla.f32   q11, q4, d1[1]      \n"

                    "vmla.f32   q8, q5, d2[0]       \n"
                    "vmla.f32   q9, q5, d2[1]       \n"
                    "vmla.f32   q10, q5, d3[0]      \n"
                    "vmla.f32   q11, q5, d3[1]      \n"

                    "subs       r4, r4, #1          \n"

                    "vmla.f32   q8, q6, d4[0]       \n"
                    "vmla.f32   q9, q6, d4[1]       \n"
                    "vmla.f32   q10, q6, d5[0]      \n"
                    "vmla.f32   q11, q6, d5[1]      \n"

                    "vmla.f32   q8, q7, d6[0]       \n"
                    "vmla.f32   q9, q7, d6[1]       \n"
                    "vmla.f32   q10, q7, d7[0]      \n"
                    "vmla.f32   q11, q7, d7[1]      \n"

                    "bne        0b                  \n"

                    "1:                             \n"

                    // r4 = remain = inChannel & 3;
                    "and        r4, %12, #3         \n" 
                    "cmp        r4, #0              \n"
                    "beq        3f                  \n"

                    "2:                             \n"

                    "pld        [%4, #128]          \n"
                    "vld1.f32   {d8-d9}, [%4]!      \n"

                    "pld        [%5, #128]          \n"
                    "vld1.f32   {d0-d1}, [%5]!      \n"

                    "subs       r4, r4, #1          \n"

                    "vmla.f32   q8, q4, d0[0]       \n"
                    "vmla.f32   q9, q4, d0[1]       \n"
                    "vmla.f32   q10, q4, d1[0]      \n"
                    "vmla.f32   q11, q4, d1[1]      \n"

                    "bne        2b                  \n"

                    "3:                             \n"

                    "vst1.f32   {d16-d17}, [%0]!   \n"
                    "vst1.f32   {d18-d19}, [%1]!   \n"
                    "vst1.f32   {d20-d21}, [%2]!   \n"
                    "vst1.f32   {d22-d23}, [%3]!   \n"

                    : "=r"(destptr0), // %0
                    "=r"(destptr1), // %1
                    "=r"(destptr2), // %2
                    "=r"(destptr3), // %3
                    "=r"(src_tm_ptr),  // %4
                    "=r"(kernel0)     // %5
                    : "0"(destptr0),
                    "1"(destptr1),
                    "2"(destptr2),
                    "3"(destptr3),
                    "4"(src_tm_ptr),
                    "5"(kernel0),
                    "r"(inChannel)     // %12
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
                );
#endif

#else
                float sum0_0 = 0.f;
                float sum0_1 = 0.f;
                float sum0_2 = 0.f;
                float sum0_3 = 0.f;

                float sum1_0 = 0.f;
                float sum1_1 = 0.f;
                float sum1_2 = 0.f;
                float sum1_3 = 0.f;

                float sum2_0 = 0.f;
                float sum2_1 = 0.f;
                float sum2_2 = 0.f;
                float sum2_3 = 0.f;

                float sum3_0 = 0.f;
                float sum3_1 = 0.f;
                float sum3_2 = 0.f;
                float sum3_3 = 0.f;

                for(int q = 0; q < inChannel; q++){
                    sum0_0 += src_tm_ptr[0] * kernel0[0];
                    sum0_1 += src_tm_ptr[1] * kernel0[0];
                    sum0_2 += src_tm_ptr[2] * kernel0[0];
                    sum0_3 += src_tm_ptr[3] * kernel0[0];

                    sum1_0 += src_tm_ptr[0] * kernel0[1];
                    sum1_1 += src_tm_ptr[1] * kernel0[1];
                    sum1_2 += src_tm_ptr[2] * kernel0[1];
                    sum1_3 += src_tm_ptr[3] * kernel0[1];

                    sum2_0 += src_tm_ptr[0] * kernel0[2];
                    sum2_1 += src_tm_ptr[1] * kernel0[2];
                    sum2_2 += src_tm_ptr[2] * kernel0[2];
                    sum2_3 += src_tm_ptr[3] * kernel0[2];

                    sum3_0 += src_tm_ptr[0] * kernel0[3];
                    sum3_1 += src_tm_ptr[1] * kernel0[3];
                    sum3_2 += src_tm_ptr[2] * kernel0[3];
                    sum3_3 += src_tm_ptr[3] * kernel0[3];

                    src_tm_ptr += 4；
                    kernel0 += 4;
                }

                destptr0[0] = sum0_0;
                destptr0[1] = sum0_1;
                destptr0[2] = sum0_2;
                destptr0[3] = sum0_3;

                destptr1[0] = sum1_0;
                destptr1[1] = sum1_1;
                destptr1[2] = sum1_2;
                destptr1[3] = sum1_3;

                destptr2[0] = sum2_0;
                destptr2[1] = sum2_1;
                destptr2[2] = sum2_2;
                destptr2[3] = sum2_3;

                destptr3[0] = sum3_0;
                destptr3[1] = sum3_1;
                destptr3[2] = sum3_2;
                destptr3[3] = sum3_3;

                destptr0 += 4;
                destptr1 += 4;
                destptr2 += 4;
                destptr3 += 4;
#endif
            }

            for(; i < outSize; i++){
                const float *src_tm_ptr = src_tm + ((i / 8) + (i % 8) / 4 + i % 4) * src_tm_size;
                const float *kernel0 = kernel + (c / 4) *  kernelSize;

#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    "veor       q0, q0, q0          \n"
                    "vdup.f32   q8, d0[0]           \n"
                    "vdup.f32   q9, d0[0]           \n"
                    "vdup.f32   q10, d0[0]          \n"
                    "vdup.f32   q11, d0[0]          \n"

                    // r4 = nn = inChannel >> 2
                    "lsr        r4, %12, #2         \n" 
                    "cmp        r4, #0              \n"
                    "beq        1f                  \n"

                    "0:                             \n"
                    "pld        [%4, #128]          \n"
                    "vld1.f32   {d8-d9}, [%4]!      \n"

                    "pld        [%5, #512]          \n"
                    "vldm       %5!, {d0-d7}        \n"

                    "vmla.f32   q8, q0, d8[0]       \n"
                    "vmla.f32   q9, q1, d8[1]       \n"
                    "vmla.f32   q10, q2, d9[0]      \n"
                    "vmla.f32   q11, q3, d9[1]      \n"

                    "subs       r4, r4, #1          \n"

                    "bne        0b                  \n"

                    // sum0 = q8 = [a1, b1, c1, d1]
                    // sum1 = q9 = [a2, b2, c2, d2]
                    // sum3 = q10 = [a3, b3, c3, d3]
                    // sum4 = q11 = [a4, b4, c4, d4]
                    
                    // q8 = [a1+b1,c1+d1, a2+b2, c2+d2]
                    "vadd.f32   q8, q8, q9          \n"
                    // q10 = [a3+b3, c3+d3, a4+b4, c4+d4]
                    "vadd.f32   q10, q10, q11       \n"
                    // q8 = [a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3, a4+b4+c4+d4]
                    "vadd.f32   q8, q8, q10         \n"

                    "1:                             \n"

                    // r4 = remain = inChannel & 3;
                    "and        r4, %12, #3         \n" 
                    "cmp        r4, #0              \n"
                    "beq        3f                  \n"

                    "2:                             \n"

                    "pld        [%4, #32]           \n"
                    "vld1.f32   {d8[],d9[]}, [%4]!  \n"

                    "pld        [%5, #128]          \n"
                    "vld1.f32   {d0-d1}, [%5]!      \n"

                    "subs       r4, r4, #1          \n"

                    "vmla.f32   q8, q4, q0         \n"

                    "bne        2b                  \n"

                    "3:                             \n"

                    "vst1.f32   {d16[0]}, [%0]!     \n"
                    "vst1.f32   {d16[1]}, [%1]!     \n"
                    "vst1.f32   {d17[0]}, [%2]!     \n"
                    "vst1.f32   {d17[1]}, [%3]!     \n"


                    : "=r"(destptr0), // %0
                    "=r"(destptr1), // %1
                    "=r"(destptr2), // %2
                    "=r"(destptr3), // %3
                    "=r"(src_tm_ptr),  // %4
                    "=r"(kernel0)     // %5
                    : "0"(destptr0),
                    "1"(destptr1),
                    "2"(destptr2),
                    "3"(destptr3),
                    "4"(src_tm_ptr),
                    "5"(kernel0),
                    "r"(inChannel)     // %12
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
                );
#endif

#else
                float sum0 = 0.f;
                float sum1 = 0.f;
                float sum2 = 0.f;
                float sum3 = 0.f;

                for(int q = 0; q < inChannel; q++){
                    sum0 += src_tm_ptr[0] * kernel0[0];
                    sum1 += src_tm_ptr[0] * kernel0[1];
                    sum2 += src_tm_ptr[0] * kernel0[2];
                    sum3 += src_tm_ptr[0] * kernel0[3];

                    src_tm_ptr++;
                    kernel0 += 4;
                }

                destptr0[0] = sum0;
                destptr1[0] = sum1;
                destptr2[0] = sum2;
                destptr3[0] = sum3;

                destptr0 ++;
                destptr1 ++;
                destptr2 ++;
                destptr3 ++;
#endif
            }

        }

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif    
        for(int cc = remainOutChannel; cc < outChannel; cc++){
            int c = cc;
            float *destptr0 = dest + c * outSize;

            int i = 0;
            for(; i + 7 < outSize; i += 8){
                const float *src_tm_ptr = src_tm + (i / 8) * src_tm_size;

                const float *kernel0 = kernel + (c / 4 + c % 4) *  kernelSize;

#if USE_ARM

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    
                );
#endif

#else
                float sum0 = 0.f;
                float sum1 = 0.f;
                float sum2 = 0.f;
                float sum3 = 0.f;
                float sum4 = 0.f;
                float sum5 = 0.f;
                float sum6 = 0.f;
                float sum7 = 0.f;

                for(int q = 0; q < inChannel; q++){
                    sum0 += src_tm_ptr[0] * kernel0[0];
                    sum1 += src_tm_ptr[1] * kernel0[0];
                    sum2 += src_tm_ptr[2] * kernel0[0];
                    sum3 += src_tm_ptr[3] * kernel0[0];
                    sum4 += src_tm_ptr[4] * kernel0[0];
                    sum5 += src_tm_ptr[5] * kernel0[0];
                    sum6 += src_tm_ptr[6] * kernel0[0];
                    sum7 += src_tm_ptr[7] * kernel0[0];

                    src_tm_ptr += 8;
                    kernel0++;
                }

                destptr0[0] = sum0;
                destptr0[1] = sum1;
                destptr0[2] = sum2;
                destptr0[3] = sum3;
                destptr0[4] = sum4;
                destptr0[5] = sum5;
                destptr0[6] = sum6;
                destptr0[7] = sum7;

                destptr0 += 8;
#endif
            }

            for(; i + 3 < outSize; i += 4){
                const float *src_tm_ptr = src_tm + (i / 8 + (i % 8) / 4) * src_tm_size;

                const float *kernel0 = kernel + (c / 4 + c % 4) *  kernelSize;
            
#if USE_ARM

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    
                );
#endif

#else
                float sum0 = 0.f;
                float sum1 = 0.f;
                float sum2 = 0.f;
                float sum3 = 0.f;

                for(int q = 0; q < inChannel; q++){
                    sum0 += src_tm_ptr[0] * kernel0[0];
                    sum1 += src_tm_ptr[1] * kernel0[0];
                    sum2 += src_tm_ptr[2] * kernel0[0];
                    sum3 += src_tm_ptr[3] * kernel0[0];

                    src_tm_ptr += 4;
                    kernel0++;
                }

                destptr0[0] = sum0;
                destptr0[1] = sum1;
                destptr0[2] = sum2;
                destptr0[3] = sum3;

                destptr0 += 4;
#endif
            }

            for(; i < outSize; i++){
                const float *src_tm_ptr = src_tm + (i / 8 + (i % 8) / 4 + i % 4) * src_tm_size;

                const float *kernel0 = kernel + (c / 4 + c % 4) *  kernelSize;

#if USE_ARM

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    
                );
#endif

#else
                float sum0 = 0.f;

                for(int q = 0; q < inChannel; q++){
                    sum0 += src_tm_ptr[0] * kernel0[0];
                    
                    src_tm_ptr++;
                    kernel0++;
                }   

                destptr0[0] = sum0;

                destptr0++;
#endif
            }
        }


    }
}

#endif