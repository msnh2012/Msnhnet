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
            k012 = vsetq_lane_f32(0.f, k012, 3);
            k345 = vsetq_lane_f32(0.f, k345, 3);
            k678 = vsetq_lane_f32(0.f, k678, 3);
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
                    asm volatile(
                         "pld        [%3, #192]          \n"
                        // 因为每一行连续计算 4 个输出，所以连续加载 
                        // 6个数据即可，4个窗口移动步长为1，有重叠
                        // r0 原来的内存排布 [a, b, c, d, e, f]
                        // d18 -> [a, b], r19 -> [c, d], r20 -> [e, f]
                        "vld1.f32   {d18-d20}, [%3 :64] \n" //r0
                        // r0 指针移动到下一次读取起始位置也就是 e
                        "add        %3, #16             \n" 

                        // q9 = [d18, d19] = [a, b, c, d]
                        // q10 = [d20, d21] = [e, f, *, *]
                        // q11 = [b, c, d, e]
                        // q12 = [c, d, e, f]
                        // 关于 vext 见：https://community.arm.com/developer/ip-products/processors/b/processors-ip-blog/posts/coding-for-neon---part-5-rearranging-vectors
                        // 
                        "vext.32    q11, q9, q10, #1    \n" 
                        "vext.32    q12, q9, q10, #2    \n"

                        "0:                             \n"

                        // 这里计算有点巧妙
                        // 首先因为4个卷积窗口之间是部分重叠的
                        // q9 其实可以看做是4个连续窗口的第1个元素排在一起
                        // q11 可以看做是4个连续窗口的第2个元素排在一起
                        // q12 可以看做是4个连续窗口的第3个元素排在一起

                        // 原来连续4个卷积窗口对应的数据是 
                        // [a, b, c], [b, c, d], [c, d, e], [d, e, f]
                        // 现在相当于 是数据做了下重排，但是重排的方式很巧妙
                        // q9 = [a, b, c, d]
                        // q11 = [b, c, d, e]
                        // q12 = [c, d, e, f]

                        // 然后下面的代码就很直观了，q9 和 k012 权值第1个权值相乘
                        // 因为 4 个窗口的第1个元素就只和 k012 第1个权值相乘
                        // %14 指 k012，假设 %14 放 q0 寄存器，%e 表示取 d0, %f指取 d1
                        "vmul.f32   q7, q9, %e14[0]     \n" //
                        // 4 个窗口的第2个元素就只和 k012 第2个权值相乘
                        "vmul.f32   q6, q11, %e14[1]    \n" //
                        // 4 个窗口的第3个元素就只和 k012 第3个权值相乘
                        // 这样子窗口之间的计算结果就可以直接累加
                        // 然后q13相当于只算了3x3卷积第一行 1x3 卷积，中间结果
                        // 下面指令是把剩下 的 两行计算完
                        "vmul.f32   q13, q12, %f14[0]   \n" 

                        // 计算第二行
                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d18-d20}, [%4]     \n" // r1
                        "add        %4, #16             \n"

                        //把第二行的[a, b, c, d] 和 k345 的第1个权值相乘，然后累加到q7寄存器上
                        "vmla.f32   q7, q9, %e15[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"
                        //把第二行的[b, c, d, e] 和 k345 的第2个权值相乘，然后累加到q6寄存器上
                        "vmla.f32   q6, q11, %e15[1]    \n"
                        //把第三行的[c, d, e, f] 和 k345 的第3个权值相乘，然后累加到q13寄存器上
                        "vmla.f32   q13, q12, %f15[0]   \n"


                        // 为outptr2做准备，计算第二行的 [a, b, c, d, e, f] 和 k012 的乘积
                        // 把第二行的 [a, b, c, d] 和 k012的第1个权值相乘，赋值给q8寄存器
                        "vmul.f32   q8, q9, %e14[0]     \n"
                        // 把第二行的 [b, c, d, e] 和 k012的第2个权值相乘，赋值给q14寄存器
                        "vmul.f32   q14, q11, %e14[1]   \n"
                        // 把第二行的 [c, d, e, f] 和 k012的第3个权值相乘，赋值给q15寄存器
                        "vmul.f32   q15, q12, %f14[0]   \n"
                        
                        //和上面的过程完全一致，这里是针对第三行
                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d18-d20}, [%5 :64] \n" // r2
                        "add        %5, #16             \n"
                        // 把第三行的 [a, b, c, d] 和 k678 的第1个权值相乘，然后累加到q7寄存器上
                        "vmla.f32   q7, q9, %e16[0]     \n"
                        
                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        // 把第三行的 [b, c, d, e] 和 k678 的第2个权值相乘，然后累加到q6寄存器上
                        "vmla.f32   q6, q11, %e16[1]    \n"
                        // 把第三行的 [c, d, e, f] 和 k678 的第3个权值相乘，然后累加到q13寄存器上
                        "vmla.f32   q13, q12, %f16[0]   \n"

                        // 把第三行的 [a, b, c, d] 和 k345 的第1个权值相乘，然后累加到q8寄存器上
                        "vmla.f32   q8, q9, %e15[0]     \n"
                        // 把第三行的 [b, c, d, e] 和 k345 的第2个权值相乘，然后累加到q14寄存器
                        "vmla.f32   q14, q11, %e15[1]   \n"
                        // 把第三行的 [c, d, e, f] 和 k345 的第3个权值相乘，然后累加到q15寄存器
                        "vmla.f32   q15, q12, %f15[0]   \n"

                        "pld        [%6, #192]          \n"
                        "vld1.f32   {d18-d20}, [%6]     \n" // r3
                        "add        %6, #16             \n"

                        // 把第四行的 [a, b, c, d] 和 k678 的第1个权值相乘，然后累加到q8寄存器上
                        "vmla.f32   q8, q9, %e16[0]     \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        // 把第四行的 [b, c, d, e] 和 k678 的第2个权值相乘，然后累加到q14寄存器上
                        "vmla.f32   q14, q11, %e16[1]   \n"
                        // 把第四行的 [c, d, e, f] 和 k678 的第3个权值相乘，然后累加到q15寄存器上
                        "vmla.f32   q15, q12, %f16[0]   \n"

                        "vadd.f32   q7, q7, q6          \n" // 将q6和q7累加到q7上，针对的是outptr

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d18-d20}, [%3 :64] \n" // r0

                        "vadd.f32   q8, q8, q14         \n" // 将q14和q8累加到q8上，针对的是outptr2
                        "vadd.f32   q7, q7, q13         \n" // 将q13累加到q7上，针对的是outptr
                        "vadd.f32   q8, q8, q15         \n" // 将q15和q8累加到q8上，针对的是outptr2

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "add        %3, #16             \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n" // 将q7寄存器的值存储到outptr
                        "vst1.f32   {d16-d17}, [%2]!    \n" // 将q8寄存器的值存储到outptr2

                        "subs       %0, #1              \n" // nn -= 1
                        "bne        0b                  \n" // 判断条件：nn != 0

                        "sub        %3, #16             \n" // 
                        : "=r"(nn),      // %0
                        "=r"(destptr0),  // %1
                        "=r"(destptr0_next), // %2
                        "=r"(r0),      // %3
                        "=r"(r1),      // %4
                        "=r"(r2),      // %5
                        "=r"(r3)       // %6
                        : "0"(nn),
                        "1"(destptr0),
                        "2"(destptr0_next),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "6"(r3),
                        "w"(k012), // %14
                        "w"(k345), // %15
                        "w"(k678) // %16
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    )
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

                    sum1 = vsetq_lane_f32(0.f, sum1, 3);
                    sum2 = vsetq_lane_f32(0.f, sum2, 3);

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

            for(; i < outHeight; i++){
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
                if (nn > 0){
                    asm volatile(
                        "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n" // r0
                        "add        %2, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "0:                             \n"

                        "vmul.f32   q7, q8, %e10[0]     \n"

                        "vmul.f32   q13, q10, %e10[1]   \n"
                        "vmul.f32   q14, q11, %f10[0]   \n"

                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d16-d18}, [%3]     \n" // r1
                        "add        %3, #16             \n"

                        "vmla.f32   q7, q8, %e11[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q13, q10, %e11[1]   \n"
                        "vmla.f32   q14, q11, %f11[0]   \n"

                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d16-d18}, [%4]     \n" // r2
                        "add        %4, #16             \n"

                        "vmla.f32   q7, q8, %e12[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q13, q10, %e12[1]   \n"
                        "vmla.f32   q14, q11, %f12[0]   \n"

                        "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n" // r0
                        "add        %2, #16             \n"

                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q7, q7, q14         \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        "sub        %2, #16             \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2)      // %4
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(k012), // %10
                        "w"(k345), // %11
                        "w"(k678) // %12
                        : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14"
                    );
                }
#endif

#endif

                for(; remain > 0; remain--){
#if USE_NEON
                    float32x4_t r00 = vld1q_f32(r0);
                    float32x4_t r10 = vld1q_f32(r1);
                    float32x4_t r20 = vld1q_f32(r2);

                    //sum1
                    float32x4_t sum1 = vmulq_f32(r00, k012);
                    sum1 = vmlaq_f32(sum1, r10, k345);
                    sum1 = vmlaq_f32(sum1, r20, k678);

                    float32x2_t a = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    abs = vpadd_f32(a, a);

                    *destptr0 = vget_lane_f32(a, 0);
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

                    *destptr0 = sum1;
                    
#endif
                    r0++;
                    r1++;
                    r2++;
                    destptr0++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }
        }
    }
}

#endif