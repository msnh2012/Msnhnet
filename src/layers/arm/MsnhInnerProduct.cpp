#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhInnerProduct.h"

namespace Msnhnet
{
    void InnerProductArm::InnerProduct(float *const &src,  const int &inChannel,  float *const &weight, float* &dest, const int& outChannel){
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


#if USE_NEON
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);
#endif

#if USE_NEON
            int nn = inChannel >> 2;
            int remain = inChannel & 3;
#else                
            int remain = inChannel;

#endif

#if USE_NEON

            if(nn > 0){
#if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                    for(; nn > 0; nn--){
                        float32x4_t _src0 = vld1q_f32(src0);

                        float32x4_t _w0 = vld1q_f32(w0);
                        _sum0 = vmlaq_f32(_sum0, _src0, _w0);

                        float32x4_t _w1 = vld1q_f32(w1);
                        _sum1 = vmlaq_f32(_sum1, _src0, _w1);

                        float32x4_t _w2 = vld1q_f32(w2);
                        _sum2 = vmlaq_f32(_sum2, _src0, _w2);

                        float32x4_t _w3 = vld1q_f32(w3);
                        _sum3 = vmlaq_f32(_sum3, _src0, _w3);

                        src0 += 4;
                        w0 += 4;
                        w1 += 4;
                        w2 += 4;
                        w3 += 4;
                    }

                     asm volatile(
                        "0:                             \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1]!      \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d2-d3}, [%2]!      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d4-d5}, [%3]!      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d6-d7}, [%4]!      \n"

                        "pld        [%5, #128]          \n"
                        "vld1.f32   {d8-d9}, [%5]!      \n"



                        "vmla.f32   q5, q0, q1          \n"
                        "vmla.f32   q6, q0, q2          \n"
                        "vmla.f32   q7, q0, q3          \n"
                        "vmla.f32   q8, q0, q4          \n"

                        "vst1.f32   {d10-d11}, [%6]    \n"
                        "vst1.f32   {d12-d13}, [%7]    \n"
                        "vst1.f32   {d14-d15}, [%8]    \n"
                        "vst1.f32   {d16-d17}, [%9]    \n"


                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        : "=r"(nn),  // %0
                        "=r"(src0),  // %1
                        "=r"(w0),    // %2
                        "=r"(w1),    // %3
                        "=r"(w2),    // %4
                        "=r"(w3),    // %5
                        "=r"(_sum0), // %6
                        "=r"(_sum1), // %7
                        "=r"(_sum2), // %8
                        "=r"(_sum3)  // %9

                        : "0"(nn),  // %10
                        "1"(src0),  // %11
                        "2"(w0),    // %12
                        "3"(w1),    // %13
                        "4"(w2),    // %14
                        "5"(w3),    // %15
                        "6"(_sum0), // %16
                        "7"(_sum1), // %17
                        "8"(_sum2), // %18
                        "9"(_sum3)  // %19

                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8");

#endif
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

#if USE_NEON
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
#if USE_NEON
            float32x4_t _sum0 = vdupq_n_f32(0.f);
            float32x4_t _sum1 = vdupq_n_f32(0.f);
#endif

#if USE_NEON
            int nn = inChannel >> 3;
            int remain = inChannel & 7;
#else
            int remain = inChannel;
#endif

#if USE_NEON
                if(nn > 0){
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                    asm volatile(
                        "0:                             \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1]!      \n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2]!      \n"

                        "vmla.f32   q4, q0, q2          \n"
                        "vmla.f32   q5, q1, q3          \n"

                        "vst1.f32   {d8-d9}, [%3]       \n"
                        "vst1.f32   {d10-d11}, [%4]     \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        : "=r"(nn),   // %0
                        "=r"(src0),    // %1
                        "=r"(w0),    // %2
                        "=r"(_sum0), // %3
                        "=r"(_sum1) // %4

                        : "0"(nn),
                        "1"(src0),
                        "2"(w0),
                        "3"(_sum0),
                        "4"(_sum1)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5");
#endif
                }
#endif

                for(; remain > 0; remain--){
                    sum += (*src0) * (*w0);
                    src0++;
                    w0++;
                }
#if USE_NEON
                _sum0 = vaddq_f32(_sum0, _sum1);
                float32x2_t _sumss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                _sumss = vpadd_f32(_sumss, _sumss);
                sum += vget_lane_f32(_sumss, 0);
#endif
                *destptr0 = sum;

        }

    }
}
#endif
