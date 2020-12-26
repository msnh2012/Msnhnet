#ifdef USE_ARM
#include "Msnhnet/layers/arm/aarch64/MsnhConvolution3x3s1.h"
namespace Msnhnet
{
//src conv kernel
void ConvolutionalLayerArmV8_3x3s1::conv3x3s1Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
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
            float* destptr0_next = destptr0 + outWidth;
            float* destptr1_next = destptr1 + outWidth;

            const float* src0 = src + q * in_size;
            //deal four lines and get two outputs in a feature map
            const float* r0 = src0;
            const float* r1 = src0 + inWidth;
            const float* r2 = src0 + inWidth * 2;
            const float* r3 = src0 + inWidth * 3;



#if USE_NEON
            float32x4_t k012 = vld1q_f32(k0);
            float32x4_t k345 = vld1q_f32(k0 + 3);
            float32x4_t k678 = vld1q_f32(k0 + 6);

            float32x4_t k012_next = vld1q_f32(k1);
            float32x4_t k345_next = vld1q_f32(k1 + 3);
            float32x4_t k678_next = vld1q_f32(k1 + 6);
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
                if(nn > 0){
                    asm volatile(
                        // v8.4s [a0, b0, c0, d0]
                        // v9.4s [e0, f0, g0, h0]
                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%5]        \n" // r0
                        "add    %5, %5, #16                 \n"

                        // v14.4s [a3, b3, c3, d3]
                        // v15.4s [e3, f3, g3, h3]
                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v14.4s, v15.4s}, [%8]      \n" // r3
                        "add    %8, %8, #16                 \n"

                        // v8.4s [a0, b0, c0, d0]
                        // v9.4s [e0, f0, g0, h0]
                        // v10.4s [b0, c0, d0, e0]
                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        // v14.4s [a3, b3, c3, d3]
                        // v15.4s [e3, f3, g3, h3]
                        // v11.4s [c3, d3, e3, f3]
                        "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                        "0:                                 \n"

                        //sum0
                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v6.4s}, [%1]               \n"

                        //sum1
                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v7.4s}, [%2]               \n"
                        
                        // v8.4s [a0, b0, c0, d0] 只和k012的第一个元素相乘获得v6.4s
                        "fmla   v6.4s, v8.4s, %18.s[0]      \n"
                        // v8.4s [a0, b0, c0, d0] 只和k012_next的第一个元素相乘获得v7.4s
                        "fmla   v7.4s, v8.4s, %21.s[0]      \n"

                        //sum0_next
                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v12.4s}, [%3]              \n" 

                        //sum1_next
                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v13.4s}, [%4]              \n"

                        // v14.4s [a3, b3, c3, d3] 只和k678的第一个元素相乘获得v12.4s
                        "fmla   v12.4s, v14.4s, %20.s[0]    \n"
                        // v14.4s [a3, b3, c3, d3] 只和k678_next的第一个元素相乘获得v13.4s
                        "fmla   v13.4s, v14.4s, %23.s[0]    \n"

                        // v8.4s [a0, b0, c0, d0]
                        // v9.4s [e0, f0, g0, h0]
                        // v8.4s [c0, d0, e0, f0]
                        "ext    v8.16b, v8.16b, v9.16b, #8  \n"
                        // v14.4s [a3, b3, c3, d3]
                        // v15.4s [e3, f3, g3, h3]
                        // v9.4s [b3, c3, d3, e3]
                        "ext    v9.16b, v14.16b, v15.16b, #4 \n"

                        // v10.4s [b0, c0, d0, e0]只和k012的第二个元素相乘并累加到v6.4s
                        "fmla   v6.4s, v10.4s, %18.s[1]     \n"
                        // v10.4s [b0, c0, d0, e0]只和k012_next的第二个元素相乘并累加到v7.4s
                        "fmla   v7.4s, v10.4s, %21.s[1]     \n"

                        // v11.4s [c3, d3, e3, f3]只和k678的第三个元素相乘并累加到v12.4s
                        "fmla   v12.4s, v11.4s, %20.s[2]    \n"
                        // v11.4s [c3, d3, e3, f3]只和k678_next的第三个元素相乘并累加到v13.4s
                        "fmla   v13.4s, v11.4s, %23.s[2]    \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        // v14.4s [a1, b1, c1, d1]
                        // v15.4s [e1, f1, g1, h1]
                        "ld1    {v14.4s, v15.4s}, [%6]      \n" // r1
                        "add    %6, %6, #16                 \n"

                        // v8.4s [c0, d0, e0, f0]只和k012的第三个元素相乘并累加到v6.4s
                        "fmla   v6.4s, v8.4s, %18.s[2]      \n"
                        // v8.4s [c0, d0, e0, f0]只和k012_next的第三个元素相乘并累加到v7.4s
                        "fmla   v7.4s, v8.4s, %21.s[2]      \n"
                        // v9.4s [b3, c3, d3, e3] 只和k678的第二个元素相乘并累加到v12.4s
                        "fmla   v12.4s, v9.4s, %20.s[1]     \n"
                        // v9.4s [b3, c3, d3, e3] 只和k678_next的第二个元素相乘并累加到v13.4s
                        "fmla   v13.4s, v9.4s, %23.s[1]     \n"

                        // v14.4s [a1, b1, c1, d1]
                        // v15.4s [e1, f1, g1, h1]
                        // v10.4s [b1, c1, d1, e1]
                        "ext    v10.16b, v14.16b, v15.16b, #4 \n"

                        // v14.4s [a1, b1, c1, d1] 和k345的第一个元素相乘并累加到v6.4s
                        "fmla   v6.4s, v14.4s, %19.s[0]     \n"
                        // v14.4s [a1, b1, c1, d1] 和k345_next的第一个元素相乘并累加到v7.4s
                        "fmla   v7.4s, v14.4s, %22.s[0]     \n"
                        // v14.4s [a1, b1, c1, d1] 和k012的第一个元素相乘并累加到v12.4s
                        "fmla   v12.4s, v14.4s, %18.s[0]    \n"
                        // v14.4s [a1, b1, c1, d1] 和k012_next的第一个元素相乘并累加到v13.4s
                        "fmla   v13.4s, v14.4s, %21.s[0]    \n"

                        // v14.4s [a1, b1, c1, d1]
                        // v15.4s [e1, f1, g1, h1]
                        // v11.4s [c1, d1, e1, f1]
                        "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                        // v10.4s [b1, c1, d1, e1] 和k345的第二个元素相乘并累加到v6.4s
                        "fmla   v6.4s, v10.4s, %19.s[1]     \n"
                        // v10.4s [b1, c1, d1, e1] 和k345_next的第二个元素相乘并累加到v7.4s
                        "fmla   v7.4s, v10.4s, %22.s[1]     \n"
                        // v10.4s [b1, c1, d1, e1] 和k012的第二个元素相乘并累加到v12.4s
                        "fmla   v12.4s, v10.4s, %18.s[1]    \n"
                        // v10.4s [b1, c1, d1, e1] 和k012_next的第二个元素相乘并累加到v13.4s
                        "fmla   v13.4s, v10.4s, %21.s[1]    \n"

                        // v8.4s [a2, b2, c2, d2]
                        // v9.4s [e2, f2, g2, h2]
                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%7]        \n" // r2
                        "add    %7, %7, #16                 \n"

                        // v11.4s [c1, d1, e1, f1] 和k345的第三个元素相乘并累加到v6.4s
                        "fmla   v6.4s, v11.4s, %19.s[2]     \n"
                        // v11.4s [c1, d1, e1, f1] 和k345_next的第三个元素相乘并累加到v7.4s
                        "fmla   v7.4s, v11.4s, %22.s[2]     \n"
                        // v11.4s [c1, d1, e1, f1] 和k012的第三个元素相乘并累加到v12.4s
                        "fmla   v12.4s, v11.4s, %18.s[2]    \n"
                        // v11.4s [c1, d1, e1, f1] 和k012_next的第三个元素相乘并累加到v12.4s
                        "fmla   v13.4s, v11.4s, %21.s[2]    \n"

                        // v8.4s [a2, b2, c2, d2]
                        // v9.4s [e2, f2, g2, h2]
                        // v10.4s [b2, c2, d2, e2]
                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"

                        "fmla   v6.4s, v8.4s, %20.s[0]      \n"
                        "fmla   v7.4s, v8.4s, %23.s[0]      \n"
                        "fmla   v12.4s, v8.4s, %19.s[0]     \n"
                        "fmla   v13.4s, v8.4s, %22.s[0]     \n"

                        // v8.4s [a2, b2, c2, d2]
                        // v9.4s [e2, f2, g2, h2]
                        // v10.4s [c2, d2, e2, f2]
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v6.4s, v10.4s, %20.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %23.s[1]     \n"
                        "fmla   v12.4s, v10.4s, %19.s[1]    \n"
                        "fmla   v13.4s, v10.4s, %22.s[1]    \n"

                        //
                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%5]        \n" // r0
                        "add    %5, %5, #16                 \n"

                        "fmla   v6.4s, v11.4s, %20.s[2]     \n"
                        "fmla   v7.4s, v11.4s, %23.s[2]     \n"
                        "fmla   v12.4s, v11.4s, %19.s[2]    \n"
                        "fmla   v13.4s, v11.4s, %22.s[2]    \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v14.4s, v15.4s}, [%8]      \n" // r3
                        "add    %8, %8, #16                 \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"

                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%2], #16          \n"

                        "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                        "st1    {v12.4s}, [%3], #16         \n"
                        "st1    {v13.4s}, [%4], #16         \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

                        "sub    %5, %5, #16                 \n"
                        "sub    %8, %8, #16                 \n"



                        : "=r"(nn),       // %0
                        "=r"(destptr0),  // %1
                        "=r"(destptr1),  // %2
                        "=r"(destptr0_next), // %3
                        "=r"(destptr1_next), // %4
                        "=r"(r0),       // %5
                        "=r"(r1),       // %6
                        "=r"(r2),       // %7
                        "=r"(r3)        // %8
                        : "0"(nn),
                        "1"(destptr0),
                        "2"(destptr1),
                        "3"(destptr0_next),
                        "4"(destptr1_next),
                        "5"(r0),
                        "6"(r1),
                        "7"(r2),
                        "8"(r3),
                        "w"(k012), // %18
                        "w"(k345), // %19
                        "w"(k678), // %20
                        "w"(k012_next), // %21
                        "w"(k345_next), // %22
                        "w"(k678_next)  // %23
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
                        );
                    
                }

#endif

                for(; remain > 0; remain--){

#if USE_NEON
                    float32x4_t r00 = vld1q_f32(r0);
                    float32x4_t r10 = vld1q_f32(r1);
                    float32x4_t r20 = vld1q_f32(r2);
                    float32x4_t r30 = vld1q_f32(r3);

                    //conv output1->chanel q output1 
                    float32x4_t sum0 = vmulq_f32(r00, k012);
                    //conv output1->channel q output2
                    float32x4_t sum1 = vmulq_f32(r00, k012_next);
                    sum0 = vmlaq_f32(sum0, r10, k345);
                    sum1 = vmlaq_f32(sum1, r10, k345_next);
                    sum0 = vmlaq_f32(sum0, r20, k678);
                    sum1 = vmlaq_f32(sum1, r20, k678_next);

                    //conv output2->channel q output1
                    float32x4_t sum0next = vmulq_f32(r10, k012);
                    //conv output2->channel q output2
                    float32x4_t sum1next = vmulq_f32(r10, k012_next);
                    sum0next = vmlaq_f32(sum0next, r20, k345);
                    sum1next = vmlaq_f32(sum1next, r20, k345_next);
                    sum0next = vmlaq_f32(sum0next, r30, k678);
                    sum1next = vmlaq_f32(sum1next, r30, k678_next);
                    
                    // use *destptr0 's data repalce sum0[3]
                    sum0 = vsetq_lane_f32(*destptr0, sum0, 3);
                    sum1 = vsetq_lane_f32(*destptr1, sum1, 3);
                    sum0next = vsetq_lane_f32(*destptr0_next, sum0next, 3);
                    sum1next = vsetq_lane_f32(*destptr1_next, sum1next, 3);

                    //accumulate

                    *destptr0 = vaddvq_f32(sum0);
                    *destptr1 = vaddvq_f32(sum1);
                    *destptr0_next = vaddvq_f32(sum0next);
                    *destptr1_next = vaddvq_f32(sum1next);   


#else

                    float sum0 = 0.f;
                    float sum1 = 0.f;
                    float sum0next = 0.f;
                    float sum1next = 0.f;

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

                    //conv output1->channel q output2
                    sum1 += r0[0] * k1[0];
                    sum1 += r0[1] * k1[1];
                    sum1 += r0[2] * k1[2];
                    sum1 += r1[0] * k1[3];
                    sum1 += r1[1] * k1[4];
                    sum1 += r1[2] * k1[5];
                    sum1 += r2[0] * k1[6];
                    sum1 += r2[1] * k1[7];
                    sum1 += r2[2] * k1[8];

                    //conv output2->channel q output1
                    sum0next += r1[0] * k0[0];
                    sum0next += r1[1] * k0[1];
                    sum0next += r1[2] * k0[2];
                    sum0next += r2[0] * k0[3];
                    sum0next += r2[1] * k0[4];
                    sum0next += r2[2] * k0[5];
                    sum0next += r3[0] * k0[6];
                    sum0next += r3[1] * k0[7];
                    sum0next += r3[2] * k0[8];

                    //conv output2->channel q output2
                    sum1next += r1[0] * k1[0];
                    sum1next += r1[1] * k1[1];
                    sum1next += r1[2] * k1[2];
                    sum1next += r2[0] * k1[3];
                    sum1next += r2[1] * k1[4];
                    sum1next += r2[2] * k1[5];
                    sum1next += r3[0] * k1[6];
                    sum1next += r3[1] * k1[7];
                    sum1next += r3[2] * k1[8];

                    //sum to dest
                    *destptr0 += sum0;
                    *destptr1 += sum1;
                    *destptr0_next += sum0next;
                    *destptr1_next += sum1next;

#endif
                    //update point address
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    destptr0++;
                    destptr1++;
                    destptr0_next++;
                    destptr1_next++;
                }

                r0 += 2 + inWidth;
                r1 += 2 + inWidth;
                r2 += 2 + inWidth;
                r3 += 2 + inWidth;

                destptr0 += outWidth;
                destptr1 += outWidth;
                destptr0_next += outWidth;
                destptr1_next += outWidth;
            }
            
            //deal three lines and get one output in a feature map
            for(; i < outHeight; i++){
                
#if USE_NEON
                int nn = outWidth >> 2;
                int remain = outWidth - (nn << 2);
#else                
                int remain = outWidth;

#endif

#if USE_NEON
                if(nn > 0){
                    asm volatile(
                        "0:                                 \n"

                        // v8.4s [a0, b0, c0, d0]
                        // v9.4s [e0, f0, g0, h0]
                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%3]        \n" // r0
                        "add    %3, %3, #16                 \n"

                        // sum0
                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v6.4s}, [%1]               \n" 

                        //sum1
                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v7.4s}, [%2]               \n"

                        // v8.4s [a0, b0, c0, d0] 和k012的第一个元素相乘获得v14.4s
                        "fmul   v14.4s, v8.4s, %12.s[0]     \n"
                        // v8.4s [a0, b0, c0, d0] 和k012_next的的第一个元素相乘获得v15.4s
                        "fmul   v15.4s, v8.4s, %15.s[0]     \n"

                        // v8.4s [a0, b0, c0, d0]
                        // v9.4s [e0, f0, g0, h0]
                        // v10.4s [b0, c0, d0, e0]
                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        // v11.4s [c0, d0, e0, f0]
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        // v10.4s [b0, c0, d0, e0]和k012的第二个元素相乘并累加到v6.4s
                        "fmla   v6.4s, v10.4s, %12.s[1]     \n"
                        // v10.4s [b0, c0, d0, e0]和k012_next的第二个元素相乘并累加到v7.4s
                        "fmla   v7.4s, v10.4s, %15.s[1]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%4]        \n" // r1
                        "add    %4, %4, #16                 \n"

                        "fmla   v14.4s, v11.4s, %12.s[2]    \n"
                        "fmla   v15.4s, v11.4s, %15.s[2]    \n"

                        "fmla   v6.4s, v8.4s, %13.s[0]      \n"
                        "fmla   v7.4s, v8.4s, %16.s[0]      \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v14.4s, v10.4s, %13.s[1]    \n"
                        "fmla   v15.4s, v10.4s, %16.s[1]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%5]        \n" // r2
                        "add    %5, %5, #16                 \n"

                        "fmla   v6.4s, v11.4s, %13.s[2]     \n"
                        "fmla   v7.4s, v11.4s, %16.s[2]     \n"

                        "fmla   v14.4s, v8.4s, %14.s[0]     \n"
                        "fmla   v15.4s, v8.4s, %17.s[0]     \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v6.4s, v10.4s, %14.s[1]     \n"
                        "fmla   v7.4s, v10.4s, %17.s[1]     \n"

                        "fmla   v14.4s, v11.4s, %14.s[2]    \n"
                        "fmla   v15.4s, v11.4s, %17.s[2]    \n"

                        "fadd   v6.4s, v6.4s, v14.4s        \n"
                        "fadd   v7.4s, v7.4s, v15.4s        \n"

                        "st1    {v6.4s}, [%1], #16          \n"
                        "st1    {v7.4s}, [%2], #16          \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

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
#if USE_NEON
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

                    *destptr0 = vaddvq_f32(sum0);
                    *destptr1 = vaddvq_f32(sum1);

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
                    r0++;
                    r1++;
                    r2++;
                    destptr0++;
                    destptr1++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
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

#if USE_NEON
            float32x4_t k012 = vld1q_f32(kernel0);
            float32x4_t k345 = vld1q_f32(kernel0 + 3);
            float32x4_t k678 = vld1q_f32(kernel0 + 6);
#else
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;
#endif

            int i = 0;
            for(; i + 1 < outHeight; i += 2){
#if USE_NEON
                int nn = outWidth >> 2;
                int remain = outWidth - (nn << 2);
#else
                int remain = outWidth;
#endif

#if USE_NEON
                if(nn > 0){
                    asm volatile(
                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%3]       \n" // r0
                        "add    %3, %3, #16                 \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v7.4s}, [%1]               \n" // _sum

                        "fmla   v7.4s, v9.4s, %14.s[0]      \n"
                        "fmul   v6.4s, v11.4s, %14.s[1]     \n"
                        "fmul   v13.4s, v12.4s, %14.s[2]    \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%4]       \n" // r1
                        "add    %4, %4, #16                 \n"

                        "fmla   v7.4s, v9.4s, %15.s[0]      \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "fmla   v6.4s, v11.4s, %15.s[1]     \n"
                        "fmla   v13.4s, v12.4s, %15.s[2]    \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v8.4s}, [%2]               \n" // _sum2

                        "fmla   v8.4s, v9.4s, %14.s[0]      \n"
                        "fmul   v14.4s, v11.4s, %14.s[1]    \n"
                        "fmul   v15.4s, v12.4s, %14.s[2]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%5]       \n" // r2
                        "add    %5, %5, #16                 \n"

                        "fmla   v7.4s, v9.4s, %16.s[0]      \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "fmla   v6.4s, v11.4s, %16.s[1]     \n"
                        "fmla   v13.4s, v12.4s, %16.s[2]    \n"

                        "fmla   v8.4s, v9.4s, %15.s[0]      \n"
                        "fmla   v14.4s, v11.4s, %15.s[1]    \n"
                        "fmla   v15.4s, v12.4s, %15.s[2]    \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%6]       \n" // r3
                        "add    %6, %6, #16                 \n"

                        "fmla   v8.4s, v9.4s, %16.s[0]      \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "fmla   v14.4s, v11.4s, %16.s[1]    \n"
                        "fmla   v15.4s, v12.4s, %16.s[2]    \n"

                        "fadd   v7.4s, v7.4s, v6.4s         \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v9.4s, v10.4s}, [%3]       \n" // r0

                        "fadd   v8.4s, v8.4s, v14.4s        \n"
                        "fadd   v7.4s, v7.4s, v13.4s        \n"
                        "fadd   v8.4s, v8.4s, v15.4s        \n"

                        "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                        "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                        "add    %3, %3, #16                 \n"

                        "st1    {v7.4s}, [%1], #16          \n"
                        "st1    {v8.4s}, [%2], #16          \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

                        "sub    %3, %3, #16                 \n"
                        : "=r"(nn),      // %0
                        "=r"(destptr0),  // %1
                        "=r"(destptr1), // %2
                        "=r"(r0),      // %3
                        "=r"(r1),      // %4
                        "=r"(r2),      // %5
                        "=r"(r3)       // %6
                        : "0"(nn),
                        "1"(destptr0),
                        "2"(destptr1),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "6"(r3),
                        "w"(k012), // %14
                        "w"(k345), // %15
                        "w"(k678)  // %16
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
#endif

                for(; remain > 0; remain--){
#if USE_NEON
                    float32x4_t r00 = vld1q_f32(r0);
                    float32x4_t r10 = vld1q_f32(r1);
                    float32x4_t r20 = vld1q_f32(r2);
                    float32x4_t r30 = vld1q_f32(r3);

                    float32x4_t sum0 = vmulq_f32(r00, k012);
                    sum0 = vmlaq_f32(sum0, r10, k345);
                    sum0 = vmlaq_f32(sum0, r20, k678);

                    float32x4_t sum1 = vmulq_f32(r10, k012);
                    sum1 = vmlaq_f32(sum1, r20, k345);
                    sum1 = vmlaq_f32(sum1, r30, k678);

                    *destptr0 = vaddvq_f32(sum0);
                    *destptr1 = vaddvq_f32(sum1);
                
#else
                    float sum0 = 0;
                    float sum1 = 0;

                    //conv output1->chanel q output1 
                    sum0 += r0[0] * k0[0];
                    sum0 += r0[1] * k0[1];
                    sum0 += r0[2] * k0[2];
                    sum0 += r1[0] * k1[0];
                    sum0 += r1[1] * k1[1];
                    sum0 += r1[2] * k1[2];
                    sum0 += r2[0] * k2[0];
                    sum0 += r2[1] * k2[1];
                    sum0 += r2[2] * k2[2];

                    //conv output1->channel q output2
                    sum1 += r1[0] * k0[0];
                    sum1 += r1[1] * k0[1];
                    sum1 += r1[2] * k0[2];
                    sum1 += r2[0] * k1[0];
                    sum1 += r2[1] * k1[1];
                    sum1 += r2[2] * k1[2];
                    sum1 += r3[0] * k2[0];
                    sum1 += r3[1] * k2[1];
                    sum1 += r3[2] * k2[2];

                    *destptr0 += sum0;
                    *destptr1 += sum1;
#endif
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    destptr0++;
                    destptr1++;
                }

                r0 += 2 + inWidth;
                r1 += 2 + inWidth;
                r2 += 2 + inWidth;
                r3 += 2 + inWidth;

                destptr0 += outWidth;
                destptr1 += outWidth;
            }

            for(; i < outHeight; i++){
#if USE_NEON
                int nn = outWidth >> 2;
                int remain = outWidth - (nn << 2);
#else
                int remain = outWidth;
#endif

#if USE_NEON
                if(nn > 0){
                    asm volatile(
                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%2]        \n" // r0
                        "add    %2, %2, #16                 \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "0:                                 \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v7.4s}, [%1]               \n" // _sum

                        "fmla   v7.4s, v8.4s, %10.s[0]      \n"
                        "fmul   v13.4s, v10.4s, %10.s[1]    \n"
                        "fmul   v14.4s, v11.4s, %10.s[2]    \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%3]        \n" // r1
                        "add    %3, %3, #16                 \n"

                        "fmla   v7.4s, v8.4s, %11.s[0]      \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v13.4s, v10.4s, %11.s[1]    \n"
                        "fmla   v14.4s, v11.4s, %11.s[2]    \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%4]        \n" // r2
                        "add    %4, %4, #16                 \n"

                        "fmla   v7.4s, v8.4s, %12.s[0]      \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "fmla   v13.4s, v10.4s, %12.s[1]    \n"
                        "fmla   v14.4s, v11.4s, %12.s[2]    \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%2]        \n" // r0
                        "add    %2, %2, #16                 \n"

                        "fadd   v7.4s, v7.4s, v13.4s        \n"
                        "fadd   v7.4s, v7.4s, v14.4s        \n"

                        "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                        "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                        "st1    {v7.4s}, [%1], #16          \n"

                        "subs   %w0, %w0, #1                \n"
                        "bne    0b                          \n"

                        "sub    %2, %2, #16                 \n"
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
                        : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }

#endif

                for(; remain > 0; remain--){
                    
#if USE_NEON
                    float32x4_t r00 = vld1q_f32(r0);
                    float32x4_t r10 = vld1q_f32(r1);
                    float32x4_t r20 = vld1q_f32(r2);

                    float32x4_t sum0 = vmulq_f32(r00, k012);
                    sum0 = vmlaq_f32(sum0, r10, k345);
                    sum0 = vmlaq_f32(sum0, r20, k678);

                    sum0 = vsetq_lane_f32(*destptr0, sum0, 3);

                    *destptr0 = vaddvq_f32(sum0);

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
                    r0++;
                    r1++;
                    r2++;
                    destptr0++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }
            kernel0 += 9;
        }
    }
}

}

#endif
