#include "Msnhnet/layers/arm/MsnhConvolution3x3s1.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
//src conv kernel
void ConvolutionalLayerArm3x3s1::conv3x3s1_neon(float *const &src, const int &inw, const int &inh,  const int &inch, float *const &kernel, const int &kw, 
                        const int &kh, float* &dest, const int &outw, const int &outh, const int &outch){
    int cc_outch = outch >> 1;
    int cc_remain_outch = outch << 1;
    const int in_size = inw * inh;
    const int out_size = outw * outh;
    //deal two conv output 
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif 
    for(int cc = 0; cc < cc_outch; cc++){
        int c = cc * 2;
        //get two conv output in same time
        float *dest0 = dest + c * out_size;
        float *dest1 =  dest + (c + 1) * out_size;

        for(int j = 0; j < out_size; j++) dest0[j] = 0.f;
        for(int j = 0; j < out_size; j++) dest1[j] = 0.f;

        //two output rely on two kernel
        float *k0 = kernel + c * inch * 3 * 3;
        float *k1 = kernel + (c + 1) * inch * 3 * 3;

        for(int q = 0; q < inch; q++){
            float* destptr0 = dest0;
            float* destptr1 = dest1;
            float* destptr0_next = destptr0 + outw;
            float* destptr1_next = destptr1 + outw;

            const float* src0 = src + q * in_size;
            //deal four lines and get two outputs in a feature map
            const float* r0 = src0;
            const float* r1 = src0 + inw;
            const float* r2 = src0 + inw * 2;
            const float* r3 = src0 + inw * 3;

#if USE_NEON
            float32x4_t k012 = vld1q_f32(k0);
            float32x4_t k345 = vld1q_f32(k0 + 3);
            float32x4_t k678 = v1d1q_f32(k0 + 6);

            float32x4_t k012_next = vld1q_f32(k1);
            float32x4_t k345_next = vld1q_f32(k1 + 3);
            float32x4_t k678_next = vld1q_f32(k1 + 6);
#endif

            int i = 0;
            for(; i + 1 < outh; i += 2){
                
#if USE_NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif


#if USE_NEON
                //assembly
                if(nn > 0){

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

                r0 += 2 + inw;
                r1 += 2 + inw;
                r2 += 2 + inw;
                r3 += 2 + inw;
                destptr0 += outw;
                destptr1 += outw;
                destptr0_next += outw;
                destptr1_next += outw;
            }
            
            //deal three lines and get one output in a feature map
            for(; i < outh; i++){
                
#if USE_NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else                
                int remain = outw;

#endif

#if USE_NEON
                if(nn > 0){

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
                    r3++;
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
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif 
    for(int cc = cc_remain_outch; cc < outch; cc++){
        int c = cc;
        float *dest0 = dest + c * out_size;
        for(int j = 0; j < out_size; j++) dest[j] = 0.f;
        const float* k0 = kernel + c * inch * 3 * 3;

        for(int q = 0; q < inch; q++){
            float *destptr0 = dest;
            float *destptr1 = dest + outw;

            const float* src0 = src + q * in_size;
            //deal four lines and get two outputs in a feature map
            const float* r0 = src0;
            const float* r1 = src0 + inw;
            const float* r2 = src0 + inw * 2;
            const float* r3 = src0 + inw * 3;

#if USE_NEON
            float32x4_t k012 = vld1q_f32(k0);
            float32x4_t k345 = vld1q_f32(k0 + 3);
            float32x4_t k678 = vld1q_f32(k0 + 6);
#else
            const float* k0 = k0;
            const float* k1 = k0 + 3;
            const float* k2 = k0 + 6;
#endif

            int i = 0;
            for(; i + 1 < outh; i += 2){
#if USE_NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif

#if USE_NEON

                if(nn > 0){

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

#if __aarch64__
                    *destptr0 = vaddvq_f32(sum0);
                    *destptr1 = vaddvq_f32(sum1);
#else
                    float32x2_t _ss0 = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    float32x2_t _ss1 = vadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));

                    float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                    *destptr0 = vaddvq_f32(_ss01, 0);
                    *destptr1 = vaddvq_f32(_ss01, 1);
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

                r0 += 2 + inw;
                r1 += 2 + inw;
                r2 += 2 + inw;
                r3 += 2 + inw;

                destptr0 += outw;
                destptr1 += outw;
            }

            for(; i < outh; i++){
#if USE_NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif

#if USE_NEON

                if(nn > 0){

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
#if __aarch64__
                    *destptr0 = vaddvq_f32(sum0);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
                    _ss = vpadd_fp32(_ss, _ss);

                    *destptr0 = vget_lane_f32(_ss, 0);
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
            k0 += 9;
        }
    }
}

}