#ifdef USE_ARM
#include "Msnhnet/layers/arm/aarch64/MsnhConvolution3x3s2.h"
namespace Msnhnet
{
//src conv kernel
void ConvolutionalLayerArmV8_3x3s2::conv3x3s2Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                        float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
    int ccOutChannel = outChannel >> 1;
        int ccRemainOutChannel = ccOutChannel << 1;

        const int in_size = inWidth * inHeight;
        const int out_size = outWidth * outHeight;
        //deal two conv output 
    #if USE_OMP
        #pragma omp parallel for num_threads(OMP_THREAD)
    #endif 
        for(int cc = 0; cc < ccOutChannel; cc++){
            int c = cc << 1;
            //get two conv output in same time
            float *dest0 = dest + c * out_size;
            float *dest1 =  dest + (c + 1) * out_size;

            // for(int j = 0; j < out_size; j++) dest0[j] = 0.f;
            // for(int j = 0; j < out_size; j++) dest1[j] = 0.f;

            //two output rely on two kernel
            float *k0 = kernel + c * inChannel * 3 * 3;
            float *k1 = kernel + (c + 1) * inChannel * 3 * 3;

            for(int q = 0; q < inChannel; q++){
                float* destptr0 = dest0;
                float* destptr1 = dest1;

                const float* src0 = src + q * in_size;
                //deal four lines and get two outputs in a feature map
                const float* r0 = src0;
                const float* r1 = src0 + inWidth;
                const float* r2 = src0 + inWidth * 2;



    #if USE_ARM
                float32x4_t k012 = vld1q_f32(k0);
                float32x4_t k345 = vld1q_f32(k0 + 3);
                float32x4_t k678 = vld1q_f32(k0 + 6);

                float32x4_t k012_next = vld1q_f32(k1);
                float32x4_t k345_next = vld1q_f32(k1 + 3);
                float32x4_t k678_next = vld1q_f32(k1 + 6);
    #endif

                int i = 0;
                
                //deal three lines and get one output in a feature map
                for(; i < outHeight; i++){
                    
    #if USE_ARM
                    int nn = outWidth >> 2;
                    int remain = outWidth - (nn << 2);
    #else                
                    int remain = outWidth;

    #endif

    #if USE_ARM
                    if(nn > 0){
                        asm volatile(
                            // v8.4s [a, c, e, g]
                            // v9.4s [b, d, g, h]
                            "prfm   pldl1keep, [%3, #256]       \n"
                            "ld2    {v8.4s, v9.4s}, [%3], #32   \n" 

                            "0:                                 \n"

                            // sum0 = v6
                            "prfm   pldl1keep, [%1, #128]       \n"
                            "ld1    {v6.4s}, [%1]               \n" 

                            // v8.4s [a, c, e, g] 和 k012的第一个元素相乘得到v12.4s
                            "fmul   v12.4s, v8.4s, %12.s[0]     \n"

                            // sum1 = v7
                            "prfm   pldl1keep, [%2, #128]       \n"
                            "ld1    {v7.4s}, [%2]               \n"

                            // v8.4s [a, c, e, g] 和 k012_next的第一个元素相乘得到v13.4s
                            "fmul   v13.4s, v8.4s, %15.s[0]     \n"

                            // v10.4s [i, k, m, o]
                            // v11.4s [j, l, n, p]
                            "prfm   pldl1keep, [%3, #128]       \n"
                            "ld2    {v10.4s, v11.4s}, [%3]      \n" 

                            // v9.4s [b, d, g, h] 和k012的第二个元素相乘并累加到v6.4s
                            "fmla   v6.4s, v9.4s, %12.s[1]      \n"

                            // v8.4s [a, c, e, g]
                            // v10.4s [i, k, m, o]
                            // v14.4s [c, e, g, i]
                            "ext    v14.16b, v8.16b, v10.16b, #4\n"
                            
                            // v9.4s [b, d, g, h] 和k012_next的第二个元素相乘并累加到v7.4s
                            "fmla   v7.4s, v9.4s, %15.s[1]      \n"

                            // v8.4s [a1, c1, e1, g1]
                            // v9.4s [b1, d1, f1, h1]
                            "prfm   pldl1keep, [%4, #256]       \n"
                            "ld2    {v8.4s, v9.4s}, [%4], #32   \n" // r1

                            // v14.4s [c, e, g, i] 和k012的第三个元素相乘并累加到v12.4s
                            "fmla   v12.4s, v14.4s, %12.s[2]    \n"
                            // v14.4s [c, e, g, i] 和k012_next的第三个元素相乘并累加到v13.4s 
                            "fmla   v13.4s, v14.4s, %15.s[2]    \n"

                            // v10.4s [i1, k1, m1, o1]
                            // v11.4s [j1, l1, n1, p1]
                            "prfm   pldl1keep, [%4, #128]       \n"
                            "ld2    {v10.4s, v11.4s}, [%4]      \n"

                            // v8.4s [a1, c1, e1, g1] 和k345的第一个元素相乘并累加到v6.4s
                            "fmla   v6.4s, v8.4s, %13.s[0]      \n"
                            // v8.4s [a1, c1, e1, g1] 和k345_next的第一个元素相乘并累加到v7.4s
                            "fmla   v7.4s, v8.4s, %16.s[0]      \n"

                            // v8.4s [a1, c1, e1, g1]
                            // v10.4s [i1, k1, m1, o1]
                            // v14.4s [c1, e1, g1, i1]
                            "ext    v14.16b, v8.16b, v10.16b, #4\n"

                            // v9.4s [b1, d1, f1, h1] 和k345的第二个元素相乘并累加到v12.4s
                            "fmla   v12.4s, v9.4s, %13.s[1]     \n"
                            // v9.4s [b1, d1, f1, h1] 和k345_next的第二个元素相乘并累加到v13.4s
                            "fmla   v13.4s, v9.4s, %16.s[1]     \n"

                            // v8.4s [a2, c2, e2, g2]
                            // v9.4s [b2, d2, f2, h2]
                            "prfm   pldl1keep, [%5, #256]       \n"
                            "ld2    {v8.4s, v9.4s}, [%5], #32   \n" // r2

                            // v14.4s [c1, e1, g1, f1] 和k345的第三个元素相乘并累加到v6.4s
                            // v14.4s [c1, e1, g1, f1] 和k345_next的第三个元素相乘并累加到v7.4s
                            "fmla   v6.4s, v14.4s, %13.s[2]     \n"
                            "fmla   v7.4s, v14.4s, %16.s[2]     \n"

                            // v10.4s [i2, k2, m2, o2]
                            // v11.4s [c2, e2, g2, i2]
                            "prfm   pldl1keep, [%5, #128]       \n"
                            "ld2    {v10.4s, v11.4s}, [%5]      \n"

                            // v8.4s [a2, c2, e2, g2] 和k678的第一个元素相乘并累加到v12.4s
                            // v8.4s [a2, c2, e2, g2] 和k678_next的第一个元素相乘并累加到v13.4s
                            "fmla   v12.4s, v8.4s, %14.s[0]     \n"
                            "fmla   v13.4s, v8.4s, %17.s[0]     \n"

                            // v8.4s [a2, c2, e2, g2]
                            // v10.4s [i2, k2, m2, o2]
                            // v14.4s [c2, e2, g2, i2]
                            "ext    v14.16b, v8.16b, v10.16b, #4\n"

                            // v9.4s [b2, d2, f2, h2] 和k678的第二个元素相乘并累加到v6.4s
                            // v9.4s [b2, d2, f2, h2] 和k678_next的第二个元素相乘并累加到v7.4s
                            "fmla   v6.4s, v9.4s, %14.s[1]      \n"
                            "fmla   v7.4s, v9.4s, %17.s[1]      \n"

                            // v14.4s [c2, e2, g2, i2] 和k678的第三个元素相乘并累加到v12.4s
                            // v14.4s [c2, e2, g2, i2] 和k678_next的第三个元素相乘并累加到v13.4s
                            "fmla   v12.4s, v14.4s, %14.s[2]    \n"
                            "fmla   v13.4s, v14.4s, %17.s[2]    \n"

                            "prfm   pldl1keep, [%3, #256]       \n"
                            "ld2    {v8.4s, v9.4s}, [%3], #32   \n" // v8 v9 = r0

                            // v6.4s 和 v12.4s 统一累加到v6.4s
                            // v7.4s 和 v13.4s 统一累加到v7.4s
                            "fadd   v6.4s, v6.4s, v12.4s        \n"
                            "fadd   v7.4s, v7.4s, v13.4s        \n"

                            // nn -= 1
                            "subs   %w0, %w0, #1                \n"

                            "st1    {v6.4s}, [%1], #16          \n"
                            "st1    {v7.4s}, [%2], #16          \n"

                            "bne    0b                          \n"
                            "sub    %3, %3, #32                 \n"

                            : "=r"(nn),      // %0
                            "=r"(destptr0), // %1
                            "=r"(destptr1), // %2
                            "=r"(r0),      // %3
                            "=r"(r1),      // %4
                            "=r"(r2)       // %5
                            : "0"(nn),
                            "1"(destptr0),
                            "2"(destptr1),
                            "3"(r0),
                            "4"(r1),
                            "5"(r2),
                            "w"(k012), // %12
                            "w"(k345), // %13
                            "w"(k678), // %14
                            "w"(k012_next), // %15
                            "w"(k345_next), // %16
                            "w"(k678_next)  // %17
                            : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
    #endif

                    for(; remain > 0; remain--){
    #if USE_ARM
                        float32x4_t r00 = vld1q_f32(r0);
                        float32x4_t r10 = vld1q_f32(r1);
                        float32x4_t r20 = vld1q_f32(r2);

                        //conv output1->chanel q output1 
                        float32x4_t sum0 = vmulq_f32(r00, k012);
                        //conv output1->channel q output2
                        float32x4_t sum1 = vmulq_f32(r00, k012_next);
                        sum0 = vmlaq_f32(sum0, r10, k345);
                        sum1 = vmlaq_f32(sum1, r10, k345_next);
                        sum0 = vmlaq_f32(sum0, r20, k678);
                        sum1 = vmlaq_f32(sum1, r20, k678_next);


                        // use *destptr0 's data repalce sum0[3]
                        sum0 = vsetq_lane_f32(*destptr0, sum0, 3);
                        sum1 = vsetq_lane_f32(*destptr1, sum1, 3);

    #if __aarch64__
                        *destptr0 = vaddvq_f32(sum0);
                        *destptr1 = vaddvq_f32(sum1);
    #else
                        float32x2_t _ss0 = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                        float32x2_t _ss1 = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));

                        float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                        *destptr0 =  vget_lane_f32(_ss01, 0);
                        *destptr1 =  vget_lane_f32(_ss01, 1);     
    #endif

    #else

                        float sum0 = 0.f;
                        float sum1 = 0.f;

                        //conv output1->chanel q output1
                        sum0 += r0[0] * k0[0];
                        sum0 += r0[1] * k0[1];
                        sum0 += r0[2] * k0[2];
                        sum0 += r1[0] * k0[3];
                        sum0 += r1[1] * k0[4];
                        sum0 += r1[2] * k0[5];
                        sum0 += r2[0] * k0[6];
                        sum0 += r2[1] * k0[7];
                        sum0 += r2[2] * k0[8];

                        //conv output2->channel q output1
                        sum1 += r0[0] * k1[0];
                        sum1 += r0[1] * k1[1];
                        sum1 += r0[2] * k1[2];
                        sum1 += r1[0] * k1[3];
                        sum1 += r1[1] * k1[4];
                        sum1 += r1[2] * k1[5];
                        sum1 += r2[0] * k1[6];
                        sum1 += r2[1] * k1[7];
                        sum1 += r2[2] * k1[8];

                        //sum to dest
                        *destptr0 += sum0;
                        *destptr1 += sum1;
                        //update point address
    #endif
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        destptr0++;
                        destptr1++;
                    }

                    r0 += 2 * (inWidth - outWidth);
                    r1 += 2 * (inWidth - outWidth);
                    r2 += 2 * (inWidth - outWidth);
                }
                
                //mov conv kernel
                k0 += 9;
                k1 += 9;
            }
        }

        //deal one conv output
    #if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
    #endif 


        for(int cc = ccRemainOutChannel; cc < outChannel; cc++){

            int c = cc;
            float *dest0 = dest + c * out_size;
            for(int j = 0; j < out_size; j++) dest0[j] = 0.f;
            const float* kernel0 = kernel + c * inChannel * 3 * 3;

            for(int q = 0; q < inChannel; q++){
                float *destptr0 = dest0;
                float *destptr1 = dest0 + outWidth;

                const float* src0 = src + q * in_size;
                //deal four lines and get two outputs in a feature map
                const float* r0 = src0;
                const float* r1 = src0 + inWidth;
                const float* r2 = src0 + inWidth * 2;
                const float* r3 = src0 + inWidth * 3;

    #if USE_ARM
                float32x4_t k012 = vld1q_f32(kernel0);
                float32x4_t k345 = vld1q_f32(kernel0 + 3);
                float32x4_t k678 = vld1q_f32(kernel0 + 6);
    #else
                const float* k0 = kernel0;
                const float* k1 = kernel0 + 3;
                const float* k2 = kernel0 + 6;
    #endif

                int i = 0;
                

                for(; i < outHeight; i++){
    #if USE_ARM
                    int nn = outWidth >> 2;
                    int remain = outWidth - (nn << 2);
    #else
                    int remain = outWidth;
    #endif

    #if USE_ARM

                    if (nn > 0) {
                        asm volatile(
                            "prfm       pldl1keep, [%2, #256]          \n"
                            "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                            "0:                                        \n"

                            "prfm       pldl1keep, [%1, #128]          \n"
                            "ld1        {v0.4s}, [%1]                  \n"

                            "fmla       v0.4s,  v2.4s, %10.s[0]        \n"
                            "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                            "prfm       pldl1keep, [%2, #256]          \n"
                            "ld2        {v8.4s, v9.4s}, [%2]           \n"
                            "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                            "fmul       v11.4s, v1.4s, %10.s[2]        \n"

                            "prfm       pldl1keep, [%3, #256]          \n"
                            "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                            "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                            "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                            "prfm       pldl1keep, [%3, #256]          \n"
                            "ld2        {v8.4s, v9.4s}, [%3]           \n"
                            "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                            "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                            "prfm       pldl1keep, [%4, #256]          \n"
                            "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                            "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                            "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                            "prfm       pldl1keep, [%4, #256]          \n"
                            "ld2        {v8.4s, v9.4s}, [%4]           \n"
                            "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                            "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                            "prfm       pldl1keep, [%2, #256]          \n"
                            "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                            "fadd       v0.4s, v0.4s, v10.4s           \n"
                            "fadd       v0.4s, v0.4s, v11.4s           \n"

                            "subs       %w0, %w0, #1                   \n"
                            "st1        {v0.4s}, [%1], #16             \n"
                            "bne        0b                             \n"
                            "sub        %2, %2, #32                    \n"
                            : "=r"(nn),     // %0
                            "=r"(destptr0), // %1
                            "=r"(r0),     // %2
                            "=r"(r1),     // %3
                            "=r"(r2)      // %4
                            : "0"(nn),
                            "1"(destptr0),
                            "2"(r0),
                            "3"(r1),
                            "4"(r2),
                            "w"(k012), // %10
                            "w"(k345), // %11
                            "w"(k678)  // %12
                            : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                    }
                    

    #endif

                    for(; remain > 0; remain--){
                        
    #if USE_ARM
                        float32x4_t r00 = vld1q_f32(r0);
                        float32x4_t r10 = vld1q_f32(r1);
                        float32x4_t r20 = vld1q_f32(r2);

                        float32x4_t sum0 = vmulq_f32(r00, k012);
                        sum0 = vmlaq_f32(sum0, r10, k345);
                        sum0 = vmlaq_f32(sum0, r20, k678);

                        sum0 = vsetq_lane_f32(*destptr0, sum0, 3);
    #if __aarch64__
                        *destptr0 = vaddvq_f32(sum0);
    #else
                        float32x2_t _ss0 = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                        _ss0 = vpadd_f32(_ss0, _ss0);

                        *destptr0 = vget_lane_f32(_ss0, 0);
    #endif

    #else

                        float sum0 = 0;

                        sum0 += r0[0] * k0[0];
                        sum0 += r0[1] * k0[1];
                        sum0 += r0[2] * k0[2];
                        sum0 += r1[0] * k1[0];
                        sum0 += r1[1] * k1[1];
                        sum0 += r1[2] * k1[2];
                        sum0 += r2[0] * k2[0];
                        sum0 += r2[1] * k2[1];
                        sum0 += r2[2] * k2[2];

                        *destptr0 += sum0;
    #endif
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        destptr0++;
                    }

                    r0 += 2 * (inWidth - outWidth);
                    r1 += 2 * (inWidth - outWidth);
                    r2 += 2 * (inWidth - outWidth);
                }
                kernel0 += 9;
            }
        }
}

}

#endif
