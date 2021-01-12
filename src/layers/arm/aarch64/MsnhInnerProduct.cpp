#ifdef USE_ARM
#include "Msnhnet/layers/arm/aarch64/MsnhInnerProduct.h"

namespace Msnhnet
{
    void InnerProductArm::InnerProductV8(float *const &src,  const int &inChannel,  float *const &weight, float* &dest, const int& outChannel){
        const float *weightPtr = weight;
        int ccOutChannel = 0;
        int ccRemainOutChannel = 0;

        ccOutChannel = outChannel >> 2;
        ccRemainOutChannel = ccOutChannel << 2;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = 0; cc < ccOutChannel; cc++){
            int c = cc * 4;
            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;
            float sum3 = 0.f;
            const float* w0 = weightPtr + c * inChannel;
            const float* w1 = weightPtr + (c + 1) * inChannel;
            const float* w2 = weightPtr + (c + 2) * inChannel;
            const float* w3 = weightPtr + (c + 3) * inChannel;
            float *destptr0 = dest + c;
            float *destptr1 = dest + c + 1;
            float *destptr2 = dest + c + 2;
            float *destptr3 = dest + c + 3;

            const float* src0 = src;


#if USE_ARM
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);
#endif

#if USE_ARM
            int nn = inChannel >> 2;
            int remain = inChannel & 3;
#else                
            int remain = inChannel;

#endif

#if USE_ARM

            if(nn > 0){
                asm volatile(

                    "0:                                 \n"

                    "prfm       pldl1keep, [%0, #128]   \n"
                    "ld1        {v0.4s}, [%0], #16      \n"
                    "prfm       pldl1keep, [%1, #128]   \n"
                    "ld1        {v1.4s}, [%1], #16      \n"
                    "prfm       pldl1keep, [%2, #128]   \n"
                    "ld1        {v2.4s}, [%2], #16      \n"
                    "prfm       pldl1keep, [%3, #128]   \n"
                    "ld1        {v3.4s}, [%3], #16      \n"
                    "prfm       pldl1keep, [%4, #128]   \n"
                    "ld1        {v4.4s}, [%4], #16      \n"

                    "fmla       %6.4s, v0.4s, v1.4s    \n"
                    "fmla       %7.4s, v0.4s, v2.4s    \n"
                    "fmla       %8.4s, v0.4s, v3.4s    \n"
                    "fmla       %9.4s, v0.4s, v4.4s    \n"

                    // "fmla       v6.4s, v0.4s, v1.4s    \n"
                    // "fmla       v7.4s, v0.4s, v2.4s    \n"
                    // "fmla       v8.4s, v0.4s, v3.4s    \n"
                    // "fmla       v9.4s, v0.4s, v4.4s    \n"

                    "subs       %w5, %w5, #1        \n"
                    "bne        0b                  \n"

                    // "st1        {v6.4s}, [%6]           \n"
                    // "st1        {v7.4s}, [%7]           \n"
                    // "st1        {v8.4s}, [%8]           \n"
                    // "st1        {v9.4s}, [%9]           \n"


                    : "=r"(src0),   // %0
                    "=r"(w0),       // %1
                    "=r"(w1),       // %2
                    "=r"(w2),       // %3
                    "=r"(w3),       // %4
                    "=r"(nn),       // %5
                    "=w"(_sum0),     // %6
                    "=w"(_sum1),     // %7
                    "=w"(_sum2),     // %8
                    "=w"(_sum3)      // %9
                    : "0"(src0),
                    "1"(w0),
                    "2"(w1),
                    "3"(w2),
                    "4"(w3),
                    "5"(nn),
                    "6"(_sum0),
                    "7"(_sum1),
                    "8"(_sum2),
                    "9"(_sum3)
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4"
                );
            }

#endif

            for(; remain > 0; remain--){
                sum0 += (*src0) * (*w0);
                sum1 += (*src0) * (*w1);
                sum2 += (*src0) * (*w2);
                sum3 += (*src0) * (*w3);
                src0++;
                w0++;
                w1++;
                w2++;
                w3++;
            }

#if USE_ARM
            float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
            float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
            float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
            float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sum3), vget_high_f32(_sum3));

            float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
            float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);

            sum0 += vget_lane_f32(_sum01ss, 0);
            sum1 += vget_lane_f32(_sum01ss, 1);
            sum2 += vget_lane_f32(_sum23ss, 0);
            sum3 += vget_lane_f32(_sum23ss, 1);
#endif

            *destptr0 = sum0;
            *destptr1 = sum1;
            *destptr2 = sum2;
            *destptr3 = sum3;
        }

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = ccRemainOutChannel; cc < outChannel; cc++){
            int c = cc;
            float sum = 0.f;
            const float* w0 = weightPtr + c * inChannel;
            const float *src0 = src;
            float *destptr0 = dest + c;
#if USE_ARM
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
#endif

#if USE_ARM
            int nn = inChannel >> 2;
            int remain = inChannel & 3;
#else
            int remain = inChannel;
#endif

#if USE_ARM
                if(nn > 0){
                    asm volatile(
                        "0:                                 \n"

                        "prfm       pldl1keep, [%0, #128]   \n"
                        "ld1        {v0.4s}, [%0], #16      \n"
                        "prfm       pldl1keep, [%1, #128]   \n"
                        "ld1        {v1.4s}, [%1], #16      \n"

                        "fmla       %3.4s, v0.4s, v1.4s    \n"

                        "subs       %w2, %w2, #1        \n"
                        "bne        0b                  \n"

                        : "=r"(src0),   // %0
                        "=r"(w0),       // %1
                        "=r"(nn),       // %2
                        "=w"(_sum0)     // %3
                        : "0"(src0),
                        "1"(w0),
                        "2"(nn),
                        "3"(_sum0)
                        : "cc", "memory", "v0", "v1"
                    );      
                }
#endif

                for(; remain > 0; remain--){
                    sum += (*src0) * (*w0);
                    src0++;
                    w0++;
                }
#if USE_ARM
                float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum0ss);
                sum += vget_lane_f32(_sum01ss, 0);
#endif
                *destptr0 = sum;

        }

    }
}
#endif
