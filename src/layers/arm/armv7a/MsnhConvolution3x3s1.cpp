#ifdef USE_ARM
#include "Msnhnet/layers/arm/armv7a/MsnhConvolution3x3s1.h"
namespace Msnhnet
{
//src conv kernel
void ConvolutionalLayerArm3x3s1::conv3x3s1Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
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
                int remain = outWidth - (nn << 2);
#else
                int remain = outWidth;
#endif


#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                 //assembly
                if(nn > 0){
                    asm volatile(
                        //
                        "0:                             \n"
                        // r0 [a, b, c, d, e, f]
                        "pld        [%5, #192]          \n"
						// d16 -> [a, b], d17 -> [c, d], d18 -> [e, f]
                        "vld1.f32   {d16-d18}, [%5] \n" // r0
                        //r0->e
						"add        %5, #16             \n"
						
						// r3 [a3, b3, c3, d3, e3, f3]
                        "pld        [%8, #192]          \n"
                        // d28 -> [a3, b3], d29 -> [c3, d3], d30 -> [e3, f3]
						"vld1.f32   {d28-d30}, [%8]     \n" // r3
						//r3->e3
                        "add        %8, #16             \n"
						
                        /********* conv output1->chanel q output **********/
						// q8 = [d16, d17] = [a, b, c, d]
						// q9 = [d18, d19] = [e, f, *, *]
						// q10 = [b, c, d, e]
                        "vext.32    q10, q8, q9, #1     \n"
						// q14 = [d28, d29] = [a3, b3, c3, d3]
						// q15 = [d30, d31] = [e3, f3, *, *]
						// q11 = [c3, d3, e3, f3]
                        "vext.32    q11, q14, q15, #2   \n"
						
						// sum0
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d12-d13}, [%1] \n" 

						// sum1
                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d14-d15}, [%2] \n"

                        // q8[a, b, c, d]只和k012的第一个元素相乘获得q6
                        "vmla.f32   q6, q8, %e18[0]     \n"
                        // q8[a, b, c, d]只和k012_next的第一个元素相乘获得q7
                        "vmla.f32   q7, q8, %e21[0]     \n"

						//sum0next
                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d24-d25}, [%3]     \n"
						
						//sumnext
                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d26-d27}, [%4]     \n"

                        // q14[a3, b3, c3, d3]只和k678的第一个元素相乘获得q12
                        "vmla.f32   q12, q14, %e20[0]   \n"
                        // q14[a3, b3, c3, d3]只和k678_next的第一个元素相乘获得q13
						"vmla.f32   q13, q14, %e23[0]   \n"

						// q8 = [d16, d17] = [a, b, c, d]
						// q9 = [d18, d19] = [e, f, *, *]
						// q8 = [c, d, e, f] ****
                        "vext.32    q8, q8, q9, #2      \n"
                        // q14 = [d28, d29] = [a3, b3, c3, d3]
						// q15 = [d30, d31] = [e3, f3, *, *]
						// q9  = [b3, c3, d3, e3] ****
						"vext.32    q9, q14, q15, #1    \n"
						
                        //q8[c, d, e, f]只和k012的第三个元素相乘并累加到q6
                        "vmla.f32   q6, q8, %f18[0]     \n"
                        //q8[c, d, e, f]只和k012_next的第三个元素相乘并累加到q7
                        "vmla.f32   q7, q8, %f21[0]     \n" 
                        //q9[b3, c3, d3, e3]只和k678的第二个元素相乘并累加到q12
                        "vmla.f32   q12, q9, %e20[1]    \n"
                        //q9[b3, c3, d3, e3]只和k678_next的第二个元素相乘并累加到q13
                        "vmla.f32   q13, q9, %e23[1]    \n"


                        /********* conv output2->chanel q output **********/
                        //r1 [a1, b1, c1, d1, e1, f1]
                        "pld        [%6, #192]          \n"
                        // d28 -> [a1, b1], d29 -> [c1, d1], d30 -> [e1, f1]
                        "vld1.f32   {d28-d30}, [%6]     \n" // r1
                        "add        %6, #16             \n"

                        // q10[b, c, d, e]只和k012的第二个元素相乘并累加到q6
                        "vmla.f32   q6, q10, %e18[1]    \n"
						// q10[b, c, d, e]只和k012_next的第二个元素相乘并累加到q7
                        "vmla.f32   q7, q10, %e21[1]    \n"
						// q11[c3, d3, e3, f3]只和k678的第三个元素相乘并累加到q11
                        "vmla.f32   q12, q11, %f20[0]   \n"
						// q11[c3, d3, e3, f3]只和k678_next的第三个元素相乘并累加到q13
                        "vmla.f32   q13, q11, %f23[0]   \n"

                        // q14 = [a1, b1, c1, d1]
                        // q15 = [e1, f1, *, *]
                        // q10 = [b1, c1, d1, e1]
                        "vext.32    q10, q14, q15, #1   \n"

                        //q14[a1, b1, c1, d1] 和 k345的第1个元素相乘并累加到q6
                        "vmla.f32   q6, q14, %e19[0]    \n"
                        //q14[a1, b1, c1, d1] 和 k345_next的第一个元素相乘并累加到q7
                        "vmla.f32   q7, q14, %e22[0]    \n"
                        //q14[a1, b1, c1, d1] 和 k012的第一个元素相乘并累加到q12
                        "vmla.f32   q12, q14, %e18[0]   \n"
                        //q14[a1, b1, c1, d1] 和 k012_next的第一个元素相乘并累加到q13
                        "vmla.f32   q13, q14, %e21[0]   \n"

                        // q14 = [a1, b1, c1, d1]
                        // q15 = [e1, f1, *, *]
                        // q11 = [c1, d1, e1, f1]
                        "vext.32    q11, q14, q15, #2   \n"

                        // q10[b1, c1, d1, e1] 和 k345的第二个元素相乘并累加到q6
                        "vmla.f32   q6, q10, %e19[1]    \n"
                        // q10[b1, c1, d1, e1] 和 k345_next的第二个元素相乘并累加到q7
                        "vmla.f32   q7, q10, %e22[1]    \n"
                        // q10[b1, c1, d1, e1] 和 k012的第二个元素相乘并累加到q12
                        "vmla.f32   q12, q10, %e18[1]   \n"
                        // q10[b1, c1, d1, e1] 和 k012_next的第二个元素相乘并累加到q13
                        "vmla.f32   q13, q10, %e21[1]   \n"

                        // r2: [a2, b2, c2, d2, e2, f2]
                        "pld        [%7, #192]          \n"
                        // d16->[a2, b2], d17->[c2, d2], d18->[e2, f2]
                        "vld1.f32   {d16-d18}, [%7] \n" // r2
                        "add        %7, #16             \n"

                        // q11[c1, d1, e1, f1] 和 k345的第三个元素相乘并累加到q6
                        "vmla.f32   q6, q11, %f19[0]    \n"
                        // q11[c1, d1, e1, f1] 和 k345_next的第三个元素相乘并累加到q7
                        "vmla.f32   q7, q11, %f22[0]    \n"
                        // q11[c1, d1, e1, f1] 和 k012的第三个元素相乘并累加到q12
                        "vmla.f32   q12, q11, %f18[0]   \n"
                        // q11[c1, d1, e1, f1] 和 k012_next的第三个元素相乘并累加到q7
                        "vmla.f32   q13, q11, %f21[0]   \n"


                        "vext.32    q10, q8, q9, #1     \n"

                        "vmla.f32   q6, q8, %e20[0]     \n"
                        "vmla.f32   q7, q8, %e23[0]     \n"
                        "vmla.f32   q12, q8, %e19[0]    \n"
                        "vmla.f32   q13, q8, %e22[0]    \n"

                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e20[1]    \n"
                        "vmla.f32   q7, q10, %e23[1]    \n"
                        "vmla.f32   q12, q10, %e19[1]   \n"
                        "vmla.f32   q13, q10, %e22[1]   \n"

                        "vmla.f32   q6, q11, %f20[0]    \n"
                        "vmla.f32   q7, q11, %f23[0]    \n"
                        "vmla.f32   q12, q11, %f19[0]   \n"
                        "vmla.f32   q13, q11, %f22[0]   \n"

                        "vst1.f32   {d12-d13}, [%1]!\n"
                        "vst1.f32   {d14-d15}, [%2]!\n"

                        "vst1.f32   {d24-d25}, [%3]!    \n"
                        "vst1.f32   {d26-d27}, [%4]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n" 

                        // OutputOperands 
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
                        //InputOperands
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
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                }

#endif

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

#if __aarch64__
                    *destptr0 = vaddvq_f32(sum0);
                    *destptr1 = vaddvq_f32(sum1);
                    *destptr0_next = vaddvq_f32(sum0next);
                    *destptr1_next = vaddvq_f32(sum1next);           
#else
                    //https://github.com/BBuf/ArmNeonOptimization/blob/master/src/boxFilterBetter.cpp
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
                    float32x2_t _ss0next = vadd_f32(vget_low_f32(sum0next), vget_high_f32(sum0next));
                    float32x2_t _ss1next = vadd_f32(vget_low_f32(sum1next), vget_high_f32(sum1next));

                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);
                    float32x2_t _ss01next = vpadd_f32(_ss0next, _ss1next);

                    *destptr0 =  vget_lane_f32(_ss01, 0);
                    *destptr1 =  vget_lane_f32(_ss01, 1);
                    *destptr0_next =  vget_lane_f32(_ss01next, 0);
                    *destptr1_next = vget_lane_f32(_ss01next, 1);      

#endif


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

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                if(nn > 0){
                  asm  volatile(
                       "0:                             \n"

                        //r0: [a, b, c, e, f, g]
                        "pld        [%3, #192]          \n"
                        // d16=[a, b], d17=[c, d], d18=[e, f]
                        "vld1.f32   {d16-d18}, [%3]     \n"
                        "add        %3, #16             \n"
                        
                        //sum0
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d12-d13}, [%1]     \n"

                        //sum1
                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d14-d15}, [%2]     \n"

                        // q8[a, b, c, d] 和 k012的第一个元素相乘得到q12
                        "vmul.f32   q12, q8, %e12[0]    \n"
                        // q8[a, b, c, d] 和 k012_next的第一个元素相乘得到q13
                        "vmul.f32   q13, q8, %e15[0]    \n"

                        // q8=[a, b, c, d]
                        // q9=[e, f, *, *]
                        // q10=[b, c, d, e]
                        "vext.32    q10, q8, q9, #1     \n"
                        // q11=[c, d, e, f]
                        "vext.32    q11, q8, q9, #2     \n"
                        // q10=[b, c, d, e]和k012的第二个元素相乘并累加到q6
                        "vmla.f32   q6, q10, %e12[1]    \n"
                        // q10=[b, c, d, e]和k012_next的第三个元素相乘并累加到q7
                        "vmla.f32   q7, q10, %e15[1]    \n"

                        //r1
                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d16-d18}, [%4]     \n" 
                        "add        %4, #16             \n"

                        // q11=[c, d, e, f]和k012的第三个元素相乘并累加到q12
                        "vmla.f32   q12, q11, %f12[0]   \n"
                        // q11=[c, d, e, f]和k012_next的第三个元素相乘并累加到q13
                        "vmla.f32   q13, q11, %f15[0]   \n"

                        "vmla.f32   q6, q8, %e13[0]     \n"
                        "vmla.f32   q7, q8, %e16[0]     \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q11, %f13[0]    \n"
                        "vmla.f32   q7, q11, %f16[0]    \n"

                        //r2
                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d16-d18}, [%5]     \n" 
                        "add        %5, #16             \n"

                        "vmla.f32   q12, q10, %e13[1]   \n"
                        "vmla.f32   q13, q10, %e16[1]   \n"

                        "vmla.f32   q12, q8, %e14[0]    \n"
                        "vmla.f32   q13, q8, %e17[0]    \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q6, q10, %e14[1]    \n"
                        "vmla.f32   q7, q10, %e17[1]    \n"

                        "vmla.f32   q12, q11, %f14[0]   \n"
                        "vmla.f32   q13, q11, %f17[0]   \n"

                        //sum1&&sum2
                        "vadd.f32   q6, q6, q12         \n"
                        "vadd.f32   q7, q7, q13         \n"

                        //store
                        "vst1.f32   {d12-d13}, [%1]!    \n"

                        "vst1.f32   {d14-d15}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

                        // OutputOperands 
                       : "=r"(nn),      // %0
                        "=r"(destptr0), // %1
                        "=r"(destptr1), // %2
                        "=r"(r0),      // %3
                        "=r"(r1),      // %4
                        "=r"(r2)       // %5
                        // InputOperands
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
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13"
                  );
                }
#endif

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

#if  __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                if(nn > 0){
                    asm volatile(
                        "0:                             \n"
                        // r0 = [a, b, c, d, e, f]
                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d18-d20}, [%3] \n" 
                        "add        %3, #16             \n"
                        
                        // q9 = [a, b, c, d]
                        // q10 = [e, f, *, *]
                        // q11 = [b, c, d, e]
                        "vext.32    q11, q9, q10, #1    \n"
                        // q12 = [c, d, e, f]
                        "vext.32    q12, q9, q10, #2    \n"

                        //sum0
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d14-d15}, [%1] \n"
                        // sum1
                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d16-d17}, [%2]     \n" 

                        //分别和k012的对应元素相乘
                        "vmla.f32   q7, q9, %e14[0]     \n"
                        "vmul.f32   q6, q11, %e14[1]    \n"
                        "vmul.f32   q13, q12, %f14[0]   \n"

                        // r1 = [a1, b1, c1, d1, e1, f1]
                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d18-d20}, [%4]     \n" 
                        "add        %4, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q7, q9, %e15[0]     \n"
                        "vmla.f32   q6, q11, %e15[1]    \n"
                        "vmla.f32   q13, q12, %f15[0]   \n"

                        "vmla.f32   q8, q9, %e14[0]     \n"
                        "vmul.f32   q14, q11, %e14[1]   \n"
                        "vmul.f32   q15, q12, %f14[0]   \n"

                        // r2 = [a2, b2, c2, d2, e2, f2]
                        "pld        [%5, #192]          \n"
                        "vld1.f32   {d18-d20}, [%5] \n" 
                        "add        %5, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q7, q9, %e16[0]     \n"
                        "vmla.f32   q6, q11, %e16[1]    \n"
                        "vmla.f32   q13, q12, %f16[0]   \n"

                        "vmla.f32   q8, q9, %e15[0]     \n"
                        "vmla.f32   q14, q11, %e15[1]   \n"
                        "vmla.f32   q15, q12, %f15[0]   \n"

                        // r3 = [a3, b3, c3, d3, e3, f3]
                        "pld        [%6, #192]          \n"
                        "vld1.f32   {d18-d20}, [%6]     \n" 
                        "add        %6, #16             \n"

                        "vext.32    q11, q9, q10, #1    \n"
                        "vext.32    q12, q9, q10, #2    \n"

                        "vmla.f32   q8, q9, %e16[0]     \n"
                        "vmla.f32   q14, q11, %e16[1]   \n"
                        "vmla.f32   q15, q12, %f16[0]   \n"

                        "vadd.f32   q7, q7, q6          \n"
                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q8, q8, q14         \n"
                        "vadd.f32   q8, q8, q15         \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"
                        "vst1.f32   {d16-d17}, [%2]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"


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
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
                } 
#endif

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

#if __aarch64__
                    *destptr0 = vaddvq_f32(sum0);
                    *destptr1 = vaddvq_f32(sum1);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));

                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                    *destptr0 = vget_lane_f32(_ss01, 0);
                    *destptr1 = vget_lane_f32(_ss01, 1);
#endif      
                
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

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else    
                if(nn > 0){
                    asm volatile(
                        "0:                             \n"
                        // r0 = [a, b, c, d, e, f]
                        "pld        [%2, #192]          \n"
                        "vld1.f32   {d16-d18}, [%2]     \n" 
                        "add        %2, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        //sum0
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d14-d15}, [%1]     \n"

                        "vmla.f32   q7, q8, %e10[0]     \n"
                        "vmul.f32   q13, q10, %e10[1]   \n"
                        "vmul.f32   q14, q11, %f10[0]   \n"

                        // r1 = [a1, b1, c1, d1, e1, f1]
                        "pld        [%3, #192]          \n"
                        "vld1.f32   {d16-d18}, [%3]     \n"
                        "add        %3, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"
                        
                        "vmla.f32   q7, q8, %e11[0]     \n"
                        "vmla.f32   q13, q10, %e11[1]   \n"
                        "vmla.f32   q14, q11, %f11[0]   \n"

                        // r2 = [a2, b2, c2, d2, e2, f2]
                        "pld        [%4, #192]          \n"
                        "vld1.f32   {d16-d18}, [%4]     \n"
                        "add        %4, #16             \n"

                        "vext.32    q10, q8, q9, #1     \n"
                        "vext.32    q11, q8, q9, #2     \n"

                        "vmla.f32   q7, q8, %e12[0]     \n"
                        "vmla.f32   q13, q10, %e12[1]   \n"
                        "vmla.f32   q14, q11, %f12[0]   \n"

                        "vadd.f32   q7, q7, q13         \n"
                        "vadd.f32   q7, q7, q14         \n"

                        "vst1.f32   {d14-d15}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"

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
                        : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                } 
#endif
                

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
