#ifdef USE_ARM
#include "Msnhnet/layers/arm/armv7a/MsnhConvolution3x3s1Winograd.h"
#include "Msnhnet/layers/arm/MsnhPadding.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    // kerneltm: [outChannel, inChannel, 8*8]
    //F(m, r) = GgG^T
void ConvolutionalLayerArm3x3s1Winograd::conv3x3s1WinogradTransformKenel(float *const &kernel, float* &kernel_tm, float* &kernel_tm2,const int &inChannel, const int &outChannel){
        // 矩阵G
        const float ktm[8][3] = {
            {1.0f,      0.0f,      0.0f},
            {-2.0f / 9, -2.0f / 9, -2.0f / 9},
            {-2.0f / 9, 2.0f / 9, -2.0f / 9},
            {1.0f / 90, 1.0f / 45, 2.0f / 45},
            {1.0f / 90, -1.0f / 45, 2.0f / 45},
            {1.0f / 45, 1.0f / 90, 1.0f / 180},
            {1.0f / 45, -1.0f / 90, 1.0f / 180},
            {0.0f, 0.0f, 1.0f}
        };

        const int kernelTmSize = inChannel * 8 * 8;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int outc = 0; outc < outChannel; outc++){
            for(int inc = 0; inc < inChannel; inc++){
                const float* kernel0 = (const float*)kernel + outc * inChannel * 9 + inc * 9;
                float *kernel_tm0 = kernel_tm + outc * kernelTmSize + inc * 64;

                //需要变换的卷积核
                const float* k0 = kernel0;
                const float* k1 = kernel0 + 3;
                const float* k2 = kernel0 + 6;

                float tmpG[8][3];    // tmp = G*g
                for(int i = 0; i < 8; i++){
                    tmpG[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                    tmpG[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                    tmpG[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                }

                //U = kernel_tm0 = G*g*G^T
                //[8*3] x [3*8]
                for(int i = 0; i < 8; i++){
                    float *tmpPtr = &tmpG[i][0];
                    for(int j = 0; j < 8; j++){
                        kernel_tm0[i * 8 + j] = tmpPtr[0] * ktm[j][0] + tmpPtr[1] * ktm[j][1] + tmpPtr[2] * ktm[j][2];
                    }
                }

            }
        }

        int nnOutchannel = outChannel >> 2;
        int remainOutChannel = nnOutchannel << 2;
        
        int packOutChannel = nnOutchannel + (outChannel % 4 + 3) / 4;
        int packOutH = 1;
        int packOutW = (8 * 8 * inChannel * 4);

        //float *kernel_tm2 = new float[packOutChannel * packOutH * packOutW];

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif       
        for(int cc = 0; cc < nnOutchannel; cc++){
            int c = cc << 2;
            float *ktm2 = kernel_tm2 + cc * packOutH * packOutW;
            
            const float *kernel0_tm = kernel_tm + c * kernelTmSize;
            const float *kernel1_tm = kernel_tm + (c + 1) * kernelTmSize;
            const float *kernel2_tm = kernel_tm + (c + 2) * kernelTmSize;
            const float *kernel3_tm = kernel_tm + (c + 3) * kernelTmSize;

            int q = 0;

            for(; q + 1 < inChannel; q += 2){
                const float *k00 = kernel0_tm + q * 64;
                const float *k01 = kernel0_tm + (q + 1) * 64;
                const float *k10 = kernel1_tm + q * 64;
                const float *k11 = kernel1_tm + (q + 1) * 64;
                const float *k20 = kernel2_tm + q * 64;
                const float *k21 = kernel2_tm + (q + 1) * 64;
                const float *k30 = kernel3_tm + q * 64;
                const float *k31 = kernel3_tm + (q + 1) * 64;

                for(int i = 0; i < 16; i++){

#if USE_NEON

#if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                    asm volatile(

                        // ktm2[0 + j] = k00[j];
                        // ktm2[4 + j] = k01[j];
                        "vld1.f32   {d0-d1}, [%1]! \n"
                        "vld1.f32   {d2-d3}, [%2]! \n"
                        "vst1.f32   {d0-d3}, [%0]! \n"

                        // ktm2[8 + j] = k10[j];
                        // ktm2[12 + j] = k11[j];
                        "vld1.f32   {d0-d1}, [%3]! \n"
                        "vld1.f32   {d2-d3}, [%4]! \n"
                        "vst1.f32   {d0-d3}, [%0]! \n"

                        // ktm2[16 + j] = k20[j];
                        // ktm2[20 + j] = k21[j];
                        "vld1.f32   {d0-d1}, [%5]! \n"
                        "vld1.f32   {d2-d3}, [%6]! \n"
                        "vst1.f32   {d0-d3}, [%0]! \n"

                        // ktm2[24 + j] = k30[j];
                        // ktm2[28 + j] = k31[j];
                        "vld1.f32   {d0-d1}, [%7]! \n"
                        "vld1.f32   {d2-d3}, [%8]! \n"
                        "vst1.f32   {d0-d3}, [%0]! \n"

                        : "=r"(ktm2), // %0
                        "=r"(k00),  // %1
                        "=r"(k01),  // %2
                        "=r"(k10),  // %3
                        "=r"(k11),  // %4
                        "=r"(k20),  // %5
                        "=r"(k21),  // %6
                        "=r"(k30),  // %7
                        "=r"(k31)   // %8

                        :"0"(ktm2),
                        "1"(k00),
                        "2"(k01),
                        "3"(k10),
                        "4"(k11),
                        "5"(k20),
                        "6"(k21),
                        "7"(k30),
                        "8"(k31)
                        : "cc", "memory", "q0", "q1");
#endif

#else

                    for(int j = 0; j < 4; j++){
                        ktm2[0 + j] = k00[j];
                        ktm2[4 + j] = k01[j];
                        ktm2[8 + j] = k10[j];
                        ktm2[12 + j] = k11[j];
                        ktm2[16 + j] = k20[j];
                        ktm2[20 + j] = k21[j];
                        ktm2[24 + j] = k30[j];
                        ktm2[28 + j] = k31[j];
                    }

                    k00 += 4;
                    k01 += 4;
                    k10 += 4;
                    k11 += 4;
                    k20 += 4;
                    k21 += 4;
                    k30 += 4;
                    k31 += 4;
                    ktm2 += 32;
#endif
                }
            }

            //inChannel方向的拖尾部分
            for(; q < inChannel; q++){
                const float *k00 = kernel0_tm + q * 64;
                const float *k10 = kernel1_tm + q * 64;
                const float *k20 = kernel2_tm + q * 64;
                const float *k30 = kernel3_tm + q * 64;

                for(int i = 0; i < 16; i++){

#if USE_NEON

#if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                    asm volatile(

                        // ktm2[0 + j] = k00[j];
                        // ktm2[4 + j] = k01[j];
                        "vld1.f32   {d0-d1}, [%1]! \n"
                        "vld1.f32   {d2-d3}, [%2]! \n"
                        "vst1.f32   {d0-d3}, [%0]! \n"

                        // ktm2[8 + j] = k10[j];
                        // ktm2[12 + j] = k11[j];
                        "vld1.f32   {d0-d1}, [%3]! \n"
                        "vld1.f32   {d2-d3}, [%4]! \n"
                        "vst1.f32   {d0-d3}, [%0]! \n"

                        : "=r"(ktm2), // %0
                        "=r"(k00),  // %1
                        "=r"(k10),  // %2
                        "=r"(k20),  // %3
                        "=r"(k30)  // %4
                        : "0"(ktm2),
                        "1"(k00),
                        "2"(k10),
                        "3"(k20),
                        "4"(k30)
                        : "cc", "memory", "q0", "q1");
#endif

#else
                    for(int j = 0; j < 4; j++){
                        ktm2[0 + j] = k00[j];
                        ktm2[4 + j] = k10[j];
                        ktm2[8 + j] = k20[j];
                        ktm2[12 + j] = k30[j];
                    }

                    k00 += 4;
                    k10 += 4;
                    k20 += 4;
                    k30 += 4;
                    ktm2 += 16;
#endif
                }
            }

        }

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif      
        for(int cc = remainOutChannel; cc < outChannel; cc++){
            float *ktm2 = kernel_tm2  + nnOutchannel * packOutH * packOutW + 8 * 8 * inChannel * (cc - remainOutChannel);

            const float* kernel0_tm = kernel_tm + cc * kernelTmSize;

            int q = 0;

            for(; q < inChannel; q++){
                const float* k00 = kernel0_tm + q * 64;
                for(int i = 0; i < 16; i++){
#if USE_NEON                    

#if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                    asm volatile(
                        // ktm2[0 + j] = k00[j];
                        "vld1.f32   {d0-d1}, [%1]! \n"
                        "vst1.f32   {d0-d1}, [%0]! \n"

                        : "=r"(ktm2), // %0
                        "=r"(k00)  // %1
                        : "0"(ktm2),
                        "1"(k00)
                        : "cc", "memory", "q0");
#endif

#else
                    for(int j = 0; j < 4; j++){
                        ktm2[j] = k00[j];
                    }
                    k00 += 4;
                    ktm2 += 4;
#endif
                }
            }
        }        

        //kernel_tm = kernel_tm2;
        //memcpy(kernel_tm, kernel_tm2, sizeof(float)*packOutChannel * packOutH * packOutW);
    }

    // F(6x6, 3x3) <=> input: 8x8, weight: 8x8
void ConvolutionalLayerArm3x3s1Winograd::conv3x3s1WinogradNeon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 const int &kHeight, const int &kWidth, float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        
        // Vc,b = B^Td_{c,b}B
        
        // 输出特征图如果长宽不够需要Padding
        int outW = (outWidth + 5) / 6 * 6;
        int outH = (outHeight + 5) / 6 * 6;

        const int W = outW + 2;
        const int H = outH + 2;
        const int Top = 0;
        const int Left = 0;
        const int Bottom = H;
        const int Right = W;
        const int PadHeight = Bottom - Top;
        const int PadWidth = Right - Left;
        const int PadSize = PadHeight * PadWidth;
        float *srcPadding = new float[PadHeight * PadWidth * inChannel];
        PaddingLayerArm now;
        now.padding(src, inWidth, inHeight, inChannel, srcPadding, 0, H - inHeight, 0, W - inWidth, 0);

        //padding(src, inWidth, inHeight, inChannel, srcPadding, 0, H - inHeight, 0, W - inWidth, 0);
        
        const int w_tm = outW / 6 * 8;
        const int h_tm = outH / 6 * 8;
        const int tiles = w_tm / 8 * h_tm / 8;

        int src_tm_channel = inChannel;
        const int src_tm_h = 16 * w_tm / 8 * h_tm / 8;
        const int src_tm_w = 4;
        
        const int src_tm_size = src_tm_h * src_tm_w;
        float *src_tm  = new float[src_tm_channel * src_tm_h * src_tm_w];
        
        // BT = 
        // ⎡1   0    -21/4    0    21/4     0    -1  0⎤
        // ⎢                                          ⎥
        // ⎢0   1      1    -17/4  -17/4    1    1   0⎥
        // ⎢                                          ⎥
        // ⎢0   -1     1    17/4   -17/4   -1    1   0⎥
        // ⎢                                          ⎥
        // ⎢0  1/2    1/4   -5/2   -5/4     2    1   0⎥
        // ⎢                                          ⎥
        // ⎢0  -1/2   1/4    5/2   -5/4    -2    1   0⎥
        // ⎢                                          ⎥
        // ⎢0   2      4    -5/2    -5     1/2   1   0⎥
        // ⎢                                          ⎥
        // ⎢0   -2     4     5/2    -5    -1/2   1   0⎥
        // ⎢                                          ⎥
        // ⎣0   -1     0    21/4     0    -21/4  0   1⎦

        //B = 
        // ⎡1	    0	    0	   0	   0	  0	    0	  0    ⎤
	    // ⎢0	    1	    -1	   1/2	   -1/2	  2	   -2	  -1   ⎥
	    // ⎢-21/4	1	    1	   1/4	   1/4	  4	    4	  0    ⎥
	    // ⎢0	    -17/4	17/4   -5/2	   5/2	  -5/2	5/2	  21/4 ⎥
	    // ⎢21/4	-17/4	-17/4  -5/4	  -5/4	  -5	-5	  0    ⎥   
	    // ⎢0	    1	    -1	   2	   2	  1/2	-1/2  -21/4⎥
	    // ⎢-1	    1	    1	   1	   1	  1	    1	  0    ⎥
	    // ⎢0	    0	    0	   0	   0	  0	    0	  1    ⎥


        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)


#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int q = 0; q < inChannel; q++){
            float *padptr = srcPadding + q * PadSize;
            float *srcptr = src_tm + q * src_tm_size;

            float tmpV[8][8];

            //tile
            for(int i = 0; i < h_tm / 8; i++){
                for(int j = 0; j < w_tm / 8; j++){

#if USE_NEON
                    const float coeff[8] = {
                        0.25f, 0.5f, -1.25f, 2.f,
                        -2.5f, 4.f, 4.25f, 5.25f
                    };
                    
                    float32x4_t coeff0 = vld1q_f32(coeff);
                    float32x4_t coeff1 = vld1q_f32(coeff + 4);

                    float *r0 = padptr + i * 6 * PadWidth + j * 6;
                    float *r1 = r0 + PadWidth;
                    float *r2 = r0 + PadWidth * 2;
                    float *r3 = r0 + PadWidth * 3;
                    //no swap intrinsic, so based on vtrnq_f32 and combile intrinsic
#if __aarch64__
                    
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
                    // some bug, armv8 temporarily not supported!
                    for(int m = 0; m + 3 < 8; m += 4){
                        float32x4_t r0_0123 = vld1q_f32(r0);
                        float32x4_t r0_4567 = vld1q_f32(r0 + 4);
                        float32x4_t r1_0123 = vld1q_f32(r1);
                        float32x4_t r1_4567 = vld1q_f32(r1 + 4);
                        float32x4_t r2_0123 = vld1q_f32(r2);
                        float32x4_t r2_4567 = vld1q_f32(r2 + 4);
                        float32x4_t r3_0123 = vld1q_f32(r3);
                        float32x4_t r3_4567 = vld1q_f32(r3 + 4);
                        // vtrn_type: 将两个输入vector的元素通过转置生成一个有两个vector的矩阵
                        // 如：src.val[0] = {1,2,3,4,5,6,7,8}
                        // src.val[1] = {9,10,11,12,13,14,15,16}
                        // dst = vtrn_u8(src.val[0], src.val[1])时，
                        // 则 dst.val[0] = {1,9, 3,11,5,13,7,15}
                        // dst.val[1] = {2,10,4,12,6,14,8,16}
                        float32x4x2_t r01_00221133 = vtrnq_f32(r0_0123, r1_0123);
                        float32x4x2_t r01_44665577 = vtrnq_f32(r0_4567, r1_4567);
                        float32x4x2_t r23_00221133 = vtrnq_f32(r2_0123, r3_0123);
                        float32x4x2_t r23_44665577 = vtrnq_f32(r2_4567, r3_4567);

                        //vcombine_type: 将两个元素类型相同的输入vector拼接成一个同类
                        //型但大小是输入vector两倍的新vector。新vector中低部分元素存放的是第一个输入vector元素。
                        float32x4_t r00 = vcombine_f32(vget_low_f32(r01_00221133.val[0]), vget_low_f32(r23_00221133.val[0]));
                        float32x4_t r11 = vcombine_f32(vget_low_f32(r01_00221133.val[1]), vget_low_f32(r23_00221133.val[1]));
                        float32x4_t r22 = vcombine_f32(vget_high_f32(r01_00221133.val[0]), vget_high_f32(r23_00221133.val[0]));
                        float32x4_t r33 = vcombine_f32(vget_high_f32(r01_00221133.val[1]), vget_high_f32(r23_00221133.val[1]));
                        float32x4_t r44 = vcombine_f32(vget_low_f32(r01_44665577.val[0]), vget_low_f32(r23_44665577.val[0]));
                        float32x4_t r55 = vcombine_f32(vget_low_f32(r01_44665577.val[1]), vget_low_f32(r23_44665577.val[1]));
                        float32x4_t r66 = vcombine_f32(vget_high_f32(r01_44665577.val[0]), vget_high_f32(r23_44665577.val[0]));
                        float32x4_t r77 = vcombine_f32(vget_high_f32(r01_44665577.val[1]), vget_high_f32(r23_44665577.val[1]));

                        float32x4_t r06m = vsubq_f32(r00, r66);
                        float32x4_t r71m = vsubq_f32(r77, r11);
                        float32x4_t r42m = vsubq_f32(r44, r22);
                        float32x4_t r35m = vsubq_f32(r33, r55);

                        // vmla_lane_type: ri = ai + bi * c[d]
                        float32x4_t t0 = vmlaq_lane_f32(r06m, r42m, vget_high_f32(coeff1), 1);
                        float32x4_t t7 = vmlaq_lane_f32(r71m, r35m, vget_high_f32(coeff1), 1);

                        vst1q_f32(&tmpV[0][m], t0);
                        vst1q_f32(&tmpV[7][m], t7);

                        float32x4_t r26m = vaddq_f32(r22, r66);
                        float32x4_t r15m = vaddq_f32(r11, r55);

                        // vmls_lane_type: ri = ai - bi * c[d]
                        float32x4_t t1_tmp = vmlsq_lane_f32(r26m, r44, vget_high_f32(coeff1), 0);
                        float32x4_t t2_tmp = vmlsq_lane_f32(r15m, r33, vget_high_f32(coeff1), 0);

                        float32x4_t t1 = vaddq_f32(t1_tmp, t2_tmp);
                        float32x4_t t2 = vsubq_f32(t1_tmp, t2_tmp);

                        vst1q_f32(&tmpV[1][m], t1);
                        vst1q_f32(&tmpV[2][m], t2);

                        float32x4_t r4c = vmulq_lane_f32(r44, vget_high_f32(coeff0), 0);
                        float32x4_t r3c = vmulq_lane_f32(r33, vget_low_f32(coeff1), 0);

                        //float t3 = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                        //float t4 = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);
                        
                        float32x4_t t3_tmp = vaddq_f32(r66, r4c);
                        t3_tmp = vmlaq_lane_f32(t3_tmp, r22, vget_low_f32(coeff0), 0);

                        float32x4_t t4_tmp = vmlaq_lane_f32(r3c, r11, vget_low_f32(coeff0), 1);
                        t4_tmp = vmlaq_lane_f32(t4_tmp, r55, vget_high_f32(coeff0), 1);

                        float32x4_t t3 = vaddq_f32(t3_tmp, t4_tmp);
                        float32x4_t t4 = vsubq_f32(t3_tmp, t4_tmp);

                        vst1q_f32(&tmpV[3][m], t3);
                        vst1q_f32(&tmpV[4][m], t4);

                        // float t5 = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                        // float t6 = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

                        float32x4_t r24c = vaddq_f32(r22, r4c);
                        float32x4_t t5 = vmlaq_lane_f32(r66, r24c, vget_low_f32(coeff1), 1);

                        float32x4_t t6 = vmlaq_lane_f32(r3c, r11, vget_high_f32(coeff0), 1);
                        t6 = vmlaq_lane_f32(t6_tmp, r55, vget_low_f32(coeff0), 1);

                        vst1q_f32(&tmpV[5][m], t5+t6);
                        vst1q_f32(&tmpV[6][m], t5-t6);

                        r0 += PadWidth * 4;
                        r1 += PadWidth * 4;
                        r2 += PadWidth * 4;
                        r3 += PadWidth * 4;
                    }

                    const float* t0 = tmpV[0];
                    const float* t1 = tmpV[1];
                    const float* t2 = tmpV[2];
                    const float* t3 = tmpV[3];

                    float* r00 = srcptr + (i * w_tm / 8 + j) * src_tm_w;
                    float* r04 = srcptr + (i * w_tm / 8 + j + tiles) * src_tm_w;
                    float* r10 = srcptr + (i * w_tm / 8 + j + tiles * 2) * src_tm_w;
                    float* r14 = srcptr + (i * w_tm / 8 + j + tiles * 3) * src_tm_w;
                    float* r20 = srcptr + (i * w_tm / 8 + j + tiles * 4) * src_tm_w;
                    float* r24 = srcptr + (i * w_tm / 8 + j + tiles * 5) * src_tm_w;
                    float* r30 = srcptr + (i * w_tm / 8 + j + tiles * 6) * src_tm_w;
                    float* r34 = srcptr + (i * w_tm / 8 + j + tiles * 7) * src_tm_w;

                    for(int m = 0; m + 3 < 8; m += 4){
                        float32x4_t t0_0123 = vld1q_f32(t0);
                        float32x4_t t0_4567 = vld1q_f32(t0 + 4);
                        float32x4_t t1_0123 = vld1q_f32(t1);
                        float32x4_t t1_4567 = vld1q_f32(t1 + 4);
                        float32x4_t t2_0123 = vld1q_f32(t2);
                        float32x4_t t2_4567 = vld1q_f32(t2 + 4);
                        float32x4_t t3_0123 = vld1q_f32(t3);
                        float32x4_t t3_4567 = vld1q_f32(t3 + 4);

                        float32x4x2_t t01_00221133 = vtrnq_f32(t0_0123, t1_0123);
                        float32x4x2_t t01_44665577 = vtrnq_f32(t0_4567, t1_4567);
                        float32x4x2_t t23_00221133 = vtrnq_f32(t2_0123, t3_0123);
                        float32x4x2_t t23_44665577 = vtrnq_f32(t2_4567, t3_4567);

                        float32x4_t t_00 = vcombine_f32(vget_low_f32(t01_00221133.val[0]), vget_low_f32(t23_00221133.val[0]));
                        float32x4_t t_11 = vcombine_f32(vget_low_f32(t01_00221133.val[1]), vget_low_f32(t23_00221133.val[1]));
                        float32x4_t t_22 = vcombine_f32(vget_high_f32(t01_00221133.val[0]), vget_high_f32(t23_00221133.val[0]));
                        float32x4_t t_33 = vcombine_f32(vget_high_f32(t01_00221133.val[1]), vget_high_f32(t23_00221133.val[1]));
                        float32x4_t t_44 = vcombine_f32(vget_low_f32(t01_44665577.val[0]), vget_low_f32(t23_44665577.val[0]));
                        float32x4_t t_55 = vcombine_f32(vget_low_f32(t01_44665577.val[1]), vget_low_f32(t23_44665577.val[1]));
                        float32x4_t t_66 = vcombine_f32(vget_high_f32(t01_44665577.val[0]), vget_high_f32(t23_44665577.val[0]));
                        float32x4_t t_77 = vcombine_f32(vget_high_f32(t01_44665577.val[1]), vget_high_f32(t23_44665577.val[1]));

                        float32x4_t t06 = vsubq_f32(t_00, t_66);
                        float32x4_t t71 = vsubq_f32(t_77, t_11);

                        float32x4_t t42 = vsubq_f32(t_44, t_22);
                        float32x4_t t35 = vsubq_f32(t_33, t_55);

                        float32x4_t t0642 = vmlaq_lane_f32(t06, t42, vget_high_f32(coeff1), 1);
                        float32x4_t t7135 = vmlaq_lane_f32(t71, t35, vget_high_f32(coeff1), 1);

                        // r00[0] = tmpVPtr[0] - tmpVPtr[6] + (tmpVPtr[4] - tmpVPtr[2]) * 5.25f;
                        r00[0] = vgetq_lane_f32(t0642, 0);
                        r10[0] = vgetq_lane_f32(t0642, 1);
                        r20[0] = vgetq_lane_f32(t0642, 2);
                        r30[0] = vgetq_lane_f32(t0642, 3);

                        // r04[3] = tmpVPtr[7] - tmpVPtr[1] + (tmpVPtr[3] - tmpVPtr[5]) * 5.25f;
                        r04[3] = vgetq_lane_f32(t7135, 0);
                        r14[3] = vgetq_lane_f32(t7135, 1);
                        r24[3] = vgetq_lane_f32(t7135, 2);
                        r34[3] = vgetq_lane_f32(t7135, 3);

                        // float t1 = (r0[2] + r0[6] - r0[4] * 4.25f);
                        // float t2 = (r0[1] + r0[5] - r0[3] * 4.25f);
                        // r00[1] = t1 + t2;
                        // r00[2] = t1 - t2;
                        float32x4_t t26 = vaddq_f32(t_22, t_66);
                        float32x4_t t15 = vaddq_f32(t_11, t_55);

                        float32x4_t tmp12a = vmlsq_lane_f32(t26, t_44, vget_high_f32(coeff1), 0);
                        float32x4_t tmp12b = vmlsq_lane_f32(t15, t_33, vget_high_f32(coeff1), 0);

                        float32x4_t r0_tm1 = vaddq_f32(tmp12a, tmp12b);
                        float32x4_t r0_tm2 = vsubq_f32(tmp12a, tmp12b);

                        r00[1] = vgetq_lane_f32(r0_tm1, 0);
                        r10[1] = vgetq_lane_f32(r0_tm1, 1);
                        r20[1] = vgetq_lane_f32(r0_tm1, 2);
                        r30[1] = vgetq_lane_f32(r0_tm1, 3);

                        r00[2] = vgetq_lane_f32(r0_tm2, 0);
                        r10[2] = vgetq_lane_f32(r0_tm2, 1);
                        r20[2] = vgetq_lane_f32(r0_tm2, 2);
                        r30[2] = vgetq_lane_f32(r0_tm2, 3);

                        // float t3 = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                        // float t4 = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);
                        // tmpV[3][m] = t3 + t4;
                        // tmpV[4][m] = t3 - t4;
                        float32x4_t t4c = vmulq_lane_f32(t_44, vget_high_f32(coeff0), 0);
                        float32x4_t t3c = vmulq_lane_f32(t_33, vget_low_f32(coeff1), 0);

                        float32x4_t tmp34a = vaddq_f32(t_66, t4c);
                        tmp34a = vmlaq_lane_f32(tmp34, t_22, vget_low_f32(coeff0), 0);
                        
                        float32x4_t tmp34b = vmlaq_lane_f32(t3c, t_11, vget_low_f32(coeff0), 1);
                        tmp34b = vmlaq_lane_f32(tmp34b, t_55, vget_high_f32(coeff0), 1);

                        float32x4_t r0_tm3 = vaddq_f32(tmp34a, tmp34b);
                        float32x4_t r0_tm4 = vsubq_f32(tmp34a, tmp34b);

                        r00[3] = vgetq_lane_f32(r0_tm3, 0);
                        r10[3] = vgetq_lane_f32(r0_tm3, 1);
                        r20[3] = vgetq_lane_f32(r0_tm3, 2);
                        r30[3] = vgetq_lane_f32(r0_tm3, 3);

                        r04[0] = vgetq_lane_f32(r0_tm4, 0);
                        r14[0] = vgetq_lane_f32(r0_tm4, 1);
                        r24[0] = vgetq_lane_f32(r0_tm4, 2);
                        r34[0] = vgetq_lane_f32(r0_tm4, 3);

                        // float t5 = (tmpVPtr[6] + (tmpVPtr[2] - tmpVPtr[4] * 1.25f) * 4.f);
                        // float t6 = (tmpVPtr[1] * 2.f - tmpVPtr[3] * 2.5f + tmpVPtr[5] * 0.5f);
                        // r04[1] = t5 + t6;
                        // r04[2] = t5 - t6;
                        float32x4_t t2c = vaddq_f32(t_22, t4c);
                        float32x4_t tmp56a = vmlaq_lane_f32(t_66, t2c, vget_low_f32(_coeff1), 1);
                        float32x4_t tmp56b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_high_f32(_coeff0), 1);
                        tmp56b = vmlaq_lane_f32(tmp56b, t_55, vget_low_f32(coeff0), 1);

                        float32x4_t r0_tm_4_1 = vaddq_f32(tmp56a, tmp56b);
                        float32x4_t r0_tm_4_2 = vsubq_f32(tmp56a, tmp56b);

                        r04[1] = vgetq_lane_f32(r0_tm_4_1, 0);
                        r14[1] = vgetq_lane_f32(r0_tm_4_1, 1);
                        r24[1] = vgetq_lane_f32(r0_tm_4_1, 2);
                        r34[1] = vgetq_lane_f32(r0_tm_4_1, 3);

                        r04[2] = vgetq_lane_f32(r0_tm_4_2, 0);
                        r14[2] = vgetq_lane_f32(r0_tm_4_2, 1);
                        r24[2] = vgetq_lane_f32(r0_tm_4_2, 2);
                        r34[2] = vgetq_lane_f32(r0_tm_4_2, 3);

                        t0 += 8 * 4;
                        t1 += 8 * 4;
                        t2 += 8 * 4;
                        t3 += 8 * 4;

                        r00 += 2 * 4 * tiles * src_tm_w;
                        r10 += 2 * 4 * tiles * src_tm_w;
                        r20 += 2 * 4 * tiles * src_tm_w;
                        r30 += 2 * 4 * tiles * src_tm_w;
                        r04 += 2 * 4 * tiles * src_tm_w;
                        r14 += 2 * 4 * tiles * src_tm_w;
                        r24 += 2 * 4 * tiles * src_tm_w;
                        r34 += 2 * 4 * tiles * src_tm_w;

                    }
#else

                    float *t0 = tmpV[0];
                    float *t1 = tmpV[1];
                    float *t2 = tmpV[2];
                    float *t3 = tmpV[3];
                    float *t4 = tmpV[4];
                    float *t5 = tmpV[5];
                    float *t6 = tmpV[6];
                    float *t7 = tmpV[7];

                    int stepw = PadWidth * 4 * 4;

                    asm volatile(
                        // 一共2个loop，直接复制一份
                        // loop1
                        // float32x4_t r0_0123 = vld1q_f32(r0);
                        // float32x4_t r0_4567 = vld1q_f32(r0 + 4);
                        "vld1.f32   {d16-d19}, [%8], %26    \n"
                        // float32x4_t r1_0123 = vld1q_f32(r1);
                        // float32x4_t r1_4567 = vld1q_f32(r1 + 4);
                        "vld1.f32   {d20-d23}, [%9], %26    \n"
                        // float32x4_t r2_0123 = vld1q_f32(r2);
                        // float32x4_t r2_4567 = vld1q_f32(r2 + 4);
                        "vld1.f32   {d24-d27}, [%10], %26   \n"
                        // float32x4_t r3_0123 = vld1q_f32(r3);
                        // float32x4_t r3_4567 = vld1q_f32(r3 + 4);
                        "vld1.f32   {d28-d31}, [%11], %26   \n"

                        // float32x4x2_t r01_00221133 = vtrnq_f32(r0_0123, r1_0123);
                        "vtrn.32    q8, q10             \n"
                        // float32x4x2_t r01_44665577 = vtrnq_f32(r0_4567, r1_4567);
                        "vtrn.32    q9, q11             \n"
                        // float32x4x2_t r23_00221133 = vtrnq_f32(r2_0123, r3_0123);
                        "vtrn.32    q12, q14            \n"
                        // float32x4x2_t r23_44665577 = vtrnq_f32(r2_4567, r3_4567);
                        "vtrn.32    q13, q15            \n"
                        
                        //  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        // q12 = 22  q13 = 66  q14 = 33  q15 = 77
                        // float32x4_t r00 = vcombine_f32(vget_low_f32(r01_00221133.val[0]), vget_low_f32(r23_00221133.val[0]));
                        // float32x4_t r11 = vcombine_f32(vget_low_f32(r01_00221133.val[1]), vget_low_f32(r23_00221133.val[1]));
                        // float32x4_t r22 = vcombine_f32(vget_high_f32(r01_00221133.val[0]), vget_high_f32(r23_00221133.val[0]));
                        // float32x4_t r33 = vcombine_f32(vget_high_f32(r01_00221133.val[1]), vget_high_f32(r23_00221133.val[1]));
                        // float32x4_t r44 = vcombine_f32(vget_low_f32(r01_44665577.val[0]), vget_low_f32(r23_44665577.val[0]));
                        // float32x4_t r55 = vcombine_f32(vget_low_f32(r01_44665577.val[1]), vget_low_f32(r23_44665577.val[1]));
                        // float32x4_t r66 = vcombine_f32(vget_high_f32(r01_44665577.val[0]), vget_high_f32(r23_44665577.val[0]));
                        // float32x4_t r77 = vcombine_f32(vget_high_f32(r01_44665577.val[1]), vget_high_f32(r23_44665577.val[1]));
                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n" 
                        "vswp       d23, d30            \n" 

                        //float32x4_t r06m = vsubq_f32(r00, r66);
                        "vsub.f32   q2, q8, q13         \n"
                        // float32x4_t r42m = vsubq_f32(r44, r22);
                        "vsub.f32   q3, q9, q12         \n"
                        // float32x4_t r26m = vaddq_f32(r22, r66);
                        "vadd.f32   q4, q12, q13        \n"
                        // float32x4_t r15m = vaddq_f32(r11, r55);
                        "vadd.f32   q5, q10, q11        \n"

                        // float32x4_t t0 = vmlaq_lane_f32(r06m, r42m, vget_high_f32(coeff1), 1);
                        "vmla.f32   q2, q3, %f25[1]     \n"

                        // float32x4_t r3c = vmulq_lane_f32(r33, vget_low_f32(coeff1), 0);
                        "vmul.f32   q7, q14, %e25[0]    \n" 
                        // float32x4_t r4c = vmulq_lane_f32(r44, vget_high_f32(coeff0), 0);
                        "vmul.f32   q6, q9, %f24[0]     \n" 
                        // float32x4_t t1_tmp = vmlsq_lane_f32(r26m, r44, vget_high_f32(coeff1), 0);
                        "vmls.f32   q4, q9, %f25[0]     \n"
                        // float32x4_t t2_tmp = vmlsq_lane_f32(r15m, r33, vget_high_f32(coeff1), 0);
                        "vmls.f32   q5, q14, %f25[0]    \n"
                        // vst1q_f32(&tmpV[0][m], t0);
                        "vst1.f32   {d4-d5}, [%0]!      \n" 

                        "vmov       q3, q7              \n" 

                        // float32x4_t t3_tmp = vaddq_f32(r66, r4c);
                        "vadd.f32   q2, q13, q6         \n" 
                        // float32x4_t t4_tmp = vmlaq_lane_f32(r3c, r11, vget_low_f32(coeff0), 1);
                        "vmla.f32   q3, q10, %e24[1]    \n"
                        // float32x4_t t1 = vaddq_f32(t1_tmp, t2_tmp);
                        "vadd.f32   q8, q4, q5          \n"
                        // float32x4_t t2 = vsubq_f32(t1_tmp, t2_tmp);
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n" 
                        // float32x4_t r24c = vaddq_f32(r22, r4c);
                        "vadd.f32   q6, q12, q6         \n"
                        // float32x4_t t6 = vmlaq_lane_f32(r3c, r11, vget_high_f32(coeff0), 1);
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n" //q4=q13=r66
                        // t3_tmp = vmlaq_lane_f32(t3_tmp, r22, vget_low_f32(coeff0), 0);
                        "vmla.f32   q2, q12, %e24[0]    \n"
                        // t4_tmp = vmlaq_lane_f32(t4_tmp, r55, vget_high_f32(coeff0), 1);
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        // vst1q_f32(&tmpV[1][m], t1);
                        "vst1.f32   {d16-d17}, [%1]!    \n" 

                        // float32x4_t t5 = vmlaq_lane_f32(r66, r24c, vget_low_f32(coeff1), 1);
                        "vmla.f32   q4, q6, %e25[1]     \n"
                        // t6 = vmlaq_lane_f32(t6_tmp, r55, vget_low_f32(coeff0), 1);
                        "vmla.f32   q5, q11, %e24[1]    \n"

                        // vst1q_f32(&tmpV[2][m], t2);
                        "vst1.f32   {d18-d19}, [%2]!    \n" 
                        // float32x4_t t3 = vaddq_f32(t3_tmp, t4_tmp);
                        "vadd.f32   q8, q2, q3          \n"
                        // float32x4_t t4 = vsubq_f32(t3_tmp, t4_tmp);
                        "vsub.f32   q9, q2, q3          \n"

                        // float32x4_t r71m = vsubq_f32(r77, r11);
                        "vsub.f32   q6, q15, q10        \n"
                        // float32x4_t r35m = vsubq_f32(r33, r55);
                        "vsub.f32   q7, q14, q11        \n"

                        // t5+t6
                        "vadd.f32   q2, q4, q5          \n"
                        // t5-t6
                        "vsub.f32   q3, q4, q5          \n"

                        // vst1q_f32(&tmpV[3][m], t3);
                        "vst1.f32   {d16-d17}, [%3]!    \n" 
                        // vst1q_f32(&tmpV[4][m], t4);
                        "vst1.f32   {d18-d19}, [%4]!    \n" 
                        // float32x4_t t7 = vmlaq_lane_f32(r71m, r35m, vget_high_f32(coeff1), 1);
                        "vmla.f32   q6, q7, %f25[1]     \n"
                        // vst1q_f32(&tmpV[5][m], t5);
                        "vst1.f32   {d4-d5}, [%5]!      \n" 
                        // vst1q_f32(&tmpV[6][m], t6);
                        "vst1.f32   {d6-d7}, [%6]!      \n"
                        // vst1q_f32(&tmpV[7][m], t7);
                        "vst1.f32   {d12-d13}, [%7]!    \n"

                        //************************************************************************//
                        
                        // loop2
                        // float32x4_t r0_0123 = vld1q_f32(r0);
                        // float32x4_t r0_4567 = vld1q_f32(r0 + 4);
                        "vld1.f32   {d16-d19}, [%8], %26    \n"
                        // float32x4_t r1_0123 = vld1q_f32(r1);
                        // float32x4_t r1_4567 = vld1q_f32(r1 + 4);
                        "vld1.f32   {d20-d23}, [%9], %26    \n"
                        // float32x4_t r2_0123 = vld1q_f32(r2);
                        // float32x4_t r2_4567 = vld1q_f32(r2 + 4);
                        "vld1.f32   {d24-d27}, [%10], %26   \n"
                        // float32x4_t r3_0123 = vld1q_f32(r3);
                        // float32x4_t r3_4567 = vld1q_f32(r3 + 4);
                        "vld1.f32   {d28-d31}, [%11], %26   \n"

                        // float32x4x2_t r01_00221133 = vtrnq_f32(r0_0123, r1_0123);
                        "vtrn.32    q8, q10             \n"
                        // float32x4x2_t r01_44665577 = vtrnq_f32(r0_4567, r1_4567);
                        "vtrn.32    q9, q11             \n"
                        // float32x4x2_t r23_00221133 = vtrnq_f32(r2_0123, r3_0123);
                        "vtrn.32    q12, q14            \n"
                        // float32x4x2_t r23_44665577 = vtrnq_f32(r2_4567, r3_4567);
                        "vtrn.32    q13, q15            \n"
                        
                        //  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        // q12 = 22  q13 = 66  q14 = 33  q15 = 77
                        // float32x4_t r00 = vcombine_f32(vget_low_f32(r01_00221133.val[0]), vget_low_f32(r23_00221133.val[0]));
                        // float32x4_t r11 = vcombine_f32(vget_low_f32(r01_00221133.val[1]), vget_low_f32(r23_00221133.val[1]));
                        // float32x4_t r22 = vcombine_f32(vget_high_f32(r01_00221133.val[0]), vget_high_f32(r23_00221133.val[0]));
                        // float32x4_t r33 = vcombine_f32(vget_high_f32(r01_00221133.val[1]), vget_high_f32(r23_00221133.val[1]));
                        // float32x4_t r44 = vcombine_f32(vget_low_f32(r01_44665577.val[0]), vget_low_f32(r23_44665577.val[0]));
                        // float32x4_t r55 = vcombine_f32(vget_low_f32(r01_44665577.val[1]), vget_low_f32(r23_44665577.val[1]));
                        // float32x4_t r66 = vcombine_f32(vget_high_f32(r01_44665577.val[0]), vget_high_f32(r23_44665577.val[0]));
                        // float32x4_t r77 = vcombine_f32(vget_high_f32(r01_44665577.val[1]), vget_high_f32(r23_44665577.val[1]));
                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n" 
                        "vswp       d23, d30            \n" 

                        //float32x4_t r06m = vsubq_f32(r00, r66);
                        "vsub.f32   q2, q8, q13         \n"
                        // float32x4_t r42m = vsubq_f32(r44, r22);
                        "vsub.f32   q3, q9, q12         \n"
                        // float32x4_t r26m = vaddq_f32(r22, r66);
                        "vadd.f32   q4, q12, q13        \n"
                        // float32x4_t r15m = vaddq_f32(r11, r55);
                        "vadd.f32   q5, q10, q11        \n"

                        // float32x4_t t0 = vmlaq_lane_f32(r06m, r42m, vget_high_f32(coeff1), 1);
                        "vmla.f32   q2, q3, %f25[1]     \n"

                        // float32x4_t r3c = vmulq_lane_f32(r33, vget_low_f32(coeff1), 0);
                        "vmul.f32   q7, q14, %e25[0]    \n" 
                        // float32x4_t r4c = vmulq_lane_f32(r44, vget_high_f32(coeff0), 0);
                        "vmul.f32   q6, q9, %f24[0]     \n" 
                        // float32x4_t t1_tmp = vmlsq_lane_f32(r26m, r44, vget_high_f32(coeff1), 0);
                        "vmls.f32   q4, q9, %f25[0]     \n"
                        // float32x4_t t2_tmp = vmlsq_lane_f32(r15m, r33, vget_high_f32(coeff1), 0);
                        "vmls.f32   q5, q14, %f25[0]    \n"
                        // vst1q_f32(&tmpV[0][m], t0);
                        "vst1.f32   {d4-d5}, [%0]!      \n" 

                        "vmov       q3, q7              \n" 

                        // float32x4_t t3_tmp = vaddq_f32(r66, r4c);
                        "vadd.f32   q2, q13, q6         \n" 
                        // float32x4_t t4_tmp = vmlaq_lane_f32(r3c, r11, vget_low_f32(coeff0), 1);
                        "vmla.f32   q3, q10, %e24[1]    \n"
                        // float32x4_t t1 = vaddq_f32(t1_tmp, t2_tmp);
                        "vadd.f32   q8, q4, q5          \n"
                        // float32x4_t t2 = vsubq_f32(t1_tmp, t2_tmp);
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n" 
                        // float32x4_t r24c = vaddq_f32(r22, r4c);
                        "vadd.f32   q6, q12, q6         \n"
                        // float32x4_t t6 = vmlaq_lane_f32(r3c, r11, vget_high_f32(coeff0), 1);
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n" //q4=q13=r66
                        // t3_tmp = vmlaq_lane_f32(t3_tmp, r22, vget_low_f32(coeff0), 0);
                        "vmla.f32   q2, q12, %e24[0]    \n"
                        // t4_tmp = vmlaq_lane_f32(t4_tmp, r55, vget_high_f32(coeff0), 1);
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        // vst1q_f32(&tmpV[1][m], t1);
                        "vst1.f32   {d16-d17}, [%1]!    \n" 

                        // float32x4_t t5 = vmlaq_lane_f32(r66, r24c, vget_low_f32(coeff1), 1);
                        "vmla.f32   q4, q6, %e25[1]     \n"
                        // t6 = vmlaq_lane_f32(t6_tmp, r55, vget_low_f32(coeff0), 1);
                        "vmla.f32   q5, q11, %e24[1]    \n"

                        // vst1q_f32(&tmpV[2][m], t2);
                        "vst1.f32   {d18-d19}, [%2]!    \n" 
                        // float32x4_t t3 = vaddq_f32(t3_tmp, t4_tmp);
                        "vadd.f32   q8, q2, q3          \n"
                        // float32x4_t t4 = vsubq_f32(t3_tmp, t4_tmp);
                        "vsub.f32   q9, q2, q3          \n"

                        // float32x4_t r71m = vsubq_f32(r77, r11);
                        "vsub.f32   q6, q15, q10        \n"
                        // float32x4_t r35m = vsubq_f32(r33, r55);
                        "vsub.f32   q7, q14, q11        \n"

                        // t5+t6
                        "vadd.f32   q2, q4, q5          \n"
                        // t5-t6
                        "vsub.f32   q3, q4, q5          \n"

                        // vst1q_f32(&tmpV[3][m], t3);
                        "vst1.f32   {d16-d17}, [%3]!    \n" 
                        // vst1q_f32(&tmpV[4][m], t4);
                        "vst1.f32   {d18-d19}, [%4]!    \n" 
                        // float32x4_t t7 = vmlaq_lane_f32(r71m, r35m, vget_high_f32(coeff1), 1);
                        "vmla.f32   q6, q7, %f25[1]     \n"
                        // vst1q_f32(&tmpV[5][m], t5);
                        "vst1.f32   {d4-d5}, [%5]!      \n" 
                        // vst1q_f32(&tmpV[6][m], t6);
                        "vst1.f32   {d6-d7}, [%6]!      \n"
                        // vst1q_f32(&tmpV[7][m], t7);
                        "vst1.f32   {d12-d13}, [%7]!    \n"

                        : "=r"(t0), // %0
                        "=r"(t1), // %1
                        "=r"(t2), // %2
                        "=r"(t3), // %3
                        "=r"(t4), // %4
                        "=r"(t5), // %5
                        "=r"(t6), // %6
                        "=r"(t7), // %7
                        "=r"(r0), // %8
                        "=r"(r1), // %9
                        "=r"(r2), // %10
                        "=r"(r3)  // %11

                        : "0"(t0),
                        "1"(t1),
                        "2"(t2),
                        "3"(t3),
                        "4"(t4),
                        "5"(t5),
                        "6"(t6),
                        "7"(t7),
                        "8"(r0),
                        "9"(r1),
                        "10"(r2),
                        "11"(r3),
                        "w"(coeff0), // %24
                        "w"(coeff1), // %25
                        "r"(stepw)    // %26

                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                    t0 = tmpV[0];
                    t1 = tmpV[1];
                    t2 = tmpV[2];
                    t3 = tmpV[3];

                    float* r00 = srcptr + (i * w_tm / 8 + j) * src_tm_w;
                    float* r04 = srcptr + (i * w_tm / 8 + j + tiles) * src_tm_w;
                    float* r10 = srcptr + (i * w_tm / 8 + j + tiles * 2) * src_tm_w;
                    float* r14 = srcptr + (i * w_tm / 8 + j + tiles * 3) * src_tm_w;
                    float* r20 = srcptr + (i * w_tm / 8 + j + tiles * 4) * src_tm_w;
                    float* r24 = srcptr + (i * w_tm / 8 + j + tiles * 5) * src_tm_w;
                    float* r30 = srcptr + (i * w_tm / 8 + j + tiles * 6) * src_tm_w;
                    float* r34 = srcptr + (i * w_tm / 8 + j + tiles * 7) * src_tm_w;

                    int step = 2 * 4 * 4 * tiles * src_tm_w;
                    
                    asm volatile(
                        // loop1
                        // float32x4_t t0_0123 = vld1q_f32(t0);
                        // float32x4_t t0_4567 = vld1q_f32(t0 + 4);
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        // t0 += 8*4 -> t0 += 32*4 = 128
                        "add        %8, %8, #128        \n"
                        // float32x4_t t1_0123 = vld1q_f32(t1);
                        // float32x4_t t1_4567 = vld1q_f32(t1 + 4);
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "add        %9, %9, #128        \n"
                        // float32x4_t t2_0123 = vld1q_f32(t2);
                        // float32x4_t t2_4567 = vld1q_f32(t2 + 4);
                        "vld1.f32   {d24-d27}, [%10]    \n"
                        "add        %10, %10, #128      \n"
                        // float32x4_t t3_0123 = vld1q_f32(t3);
                        // float32x4_t t3_4567 = vld1q_f32(t3 + 4);
                        "vld1.f32   {d28-d31}, [%11]    \n"
                        "add        %11, %11, #128      \n"

                        // float32x4x2_t t01_00221133 = vtrnq_f32(t0_0123, t1_0123);
                        "vtrn.32    q8, q10             \n"
                        // float32x4x2_t t01_44665577 = vtrnq_f32(t0_4567, t1_4567);
                        "vtrn.32    q9, q11             \n"
                        // float32x4x2_t t23_00221133 = vtrnq_f32(t2_0123, t3_0123);
                        "vtrn.32    q12, q14            \n"
                        // float32x4x2_t t23_44665577 = vtrnq_f32(t2_4567, t3_4567);
                        "vtrn.32    q13, q15            \n"

                        //  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        // q12 = 22  q13 = 66  q14 = 33  q15 = 77
                        // float32x4_t t_00 = vcombine_f32(vget_low_f32(t01_00221133.val[0]), vget_low_f32(t23_00221133.val[0]));
                        // float32x4_t t_11 = vcombine_f32(vget_low_f32(t01_00221133.val[1]), vget_low_f32(t23_00221133.val[1]));
                        // float32x4_t t_22 = vcombine_f32(vget_high_f32(t01_00221133.val[0]), vget_high_f32(t23_00221133.val[0]));
                        // float32x4_t t_33 = vcombine_f32(vget_high_f32(t01_00221133.val[1]), vget_high_f32(t23_00221133.val[1]));
                        // float32x4_t t_44 = vcombine_f32(vget_low_f32(t01_44665577.val[0]), vget_low_f32(t23_44665577.val[0]));
                        // float32x4_t t_55 = vcombine_f32(vget_low_f32(t01_44665577.val[1]), vget_low_f32(t23_44665577.val[1]));
                        // float32x4_t t_66 = vcombine_f32(vget_high_f32(t01_44665577.val[0]), vget_high_f32(t23_44665577.val[0]));
                        // float32x4_t t_77 = vcombine_f32(vget_high_f32(t01_44665577.val[1]), vget_high_f32(t23_44665577.val[1]));
                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"
                        "vswp       d23, d30            \n"

                        // float32x4_t t06 = vsubq_f32(t_00, t_66);
                        "vsub.f32   q2, q8, q13         \n"
                        // float32x4_t t42 = vsubq_f32(t_44, t_22);
                        "vsub.f32   q3, q9, q12         \n"

                        // float32x4_t t26 = vaddq_f32(t_22, t_66);
                        "vadd.f32   q4, q12, q13        \n"
                        // float32x4_t t15 = vaddq_f32(t_11, t_55);
                        "vadd.f32   q5, q10, q11        \n"
                        // float32x4_t t0642 = vmlaq_lane_f32(t06, t42, vget_high_f32(coeff1), 1);
                        "vmla.f32   q2, q3, %f25[1]     \n"
                        // float32x4_t t3c = vmulq_lane_f32(t_33, vget_low_f32(coeff1), 0);
                        "vmul.f32   q7, q14, %e25[0]    \n" 
                        // float32x4_t t4c = vmulq_lane_f32(t_44, vget_high_f32(coeff0), 0);
                        "vmul.f32   q6, q9, %f24[0]     \n" 

                        // float32x4_t tmp12a = vmlsq_lane_f32(t26, t_44, vget_high_f32(coeff1), 0);
                        "vmls.f32   q4, q9, %f25[0]     \n"
                        // float32x4_t tmp12b = vmlsq_lane_f32(t15, t_33, vget_high_f32(coeff1), 0);
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        // r00[0] = vgetq_lane_f32(t0642, 0);
                        // r10[0] = vgetq_lane_f32(t0642, 1);
                        "vst1.f32   {d4[0]}, [%0]!      \n"
                        "vst1.f32   {d4[1]}, [%2]!      \n"

                        "vmov       q3, q7              \n"  //q3 = q7

                        // r20[0] = vgetq_lane_f32(t0642, 2);
                        // r30[0] = vgetq_lane_f32(t0642, 3);
                        "vst1.f32   {d5[0]}, [%4]!      \n"
                        "vst1.f32   {d5[1]}, [%6]!      \n"
                        // float32x4_t tmp34a = vaddq_f32(t_66, t4c);
                        "vadd.f32   q2, q13, q6         \n" 
                        // float32x4_t tmp34b = vmlaq_lane_f32(t3c, t_11, vget_low_f32(coeff0), 1);
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        // float32x4_t r0_tm1 = vaddq_f32(tmp12a, tmp12b);
                        "vadd.f32   q8, q4, q5          \n"
                        // float32x4_t r0_tm2 = vsubq_f32(tmp12a, tmp12b);
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n" //q5 = q7 = t3c
                        
                        // float32x4_t t2c = vaddq_f32(t_22, t4c);
                        "vadd.f32   q6, q12, q6         \n" 
                        // float32x4_t tmp56b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_high_f32(_coeff0), 1);
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n" //q4 = q13 = t_66

                        // tmp34a = vmlaq_lane_f32(tmp34, t_22, vget_low_f32(coeff0), 0);
                        "vmla.f32   q2, q12, %e24[0]    \n"
                        // tmp34b = vmlaq_lane_f32(tmp34b, t_55, vget_high_f32(coeff0), 1);
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        // r00[1] = vgetq_lane_f32(r0_tm1, 0);
                        // r10[1] = vgetq_lane_f32(r0_tm1, 1);
                        "vst1.f32   {d16[0]}, [%0]!     \n"
                        "vst1.f32   {d16[1]}, [%2]!     \n"

                        // float32x4_t tmp56a = vmlaq_lane_f32(t_66, t2c, vget_low_f32(_coeff1), 1);
                        "vmla.f32   q4, q6, %e25[1]     \n"

                        // r20[1] = vgetq_lane_f32(r0_tm1, 2);
                        // r30[1] = vgetq_lane_f32(r0_tm1, 3);
                        "vst1.f32   {d17[0]}, [%4]!     \n"
                        "vst1.f32   {d17[1]}, [%6]!     \n"

                        // tmp56b = vmlaq_lane_f32(tmp56b, t_55, vget_low_f32(coeff0), 1);
                        "vmla.f32   q5, q11, %e24[1]    \n"

                        // r00[2] = vgetq_lane_f32(r0_tm2, 0);
                        // r10[2] = vgetq_lane_f32(r0_tm2, 1);
                        "vst1.f32   {d18[0]}, [%0]!     \n"
                        "vst1.f32   {d18[1]}, [%2]!     \n"

                        // float32x4_t r0_tm3 = vaddq_f32(tmp34a, tmp34b);
                        "vadd.f32   q8, q2, q3          \n"

                        // r20[2] = vgetq_lane_f32(r0_tm2, 2);
                        // r30[2] = vgetq_lane_f32(r0_tm2, 3);
                        "vst1.f32   {d19[0]}, [%4]!     \n"
                        "vst1.f32   {d19[1]}, [%6]!     \n"

                        // float32x4_t r0_tm4 = vsubq_f32(tmp34a, tmp34b);
                        "vsub.f32   q9, q2, q3          \n"

                        // float32x4_t t71 = vsubq_f32(t_77, t_11);
                        "vsub.f32   q6, q15, q10        \n"
                        // float32x4_t t35 = vsubq_f32(t_33, t_55);
                        "vsub.f32   q7, q14, q11        \n"

                        // float32x4_t r0_tm_4_1 = vaddq_f32(tmp56a, tmp56b);
                        "vadd.f32   q2, q4, q5          \n"
                        // float32x4_t r0_tm_4_2 = vsubq_f32(tmp56a, tmp56b);
                        "vsub.f32   q3, q4, q5          \n"

                        // r00[3] = vgetq_lane_f32(r0_tm3, 0);
                        // r10[3] = vgetq_lane_f32(r0_tm3, 1);
                        "vst1.f32   {d16[0]}, [%0], %26 \n"
                        "vst1.f32   {d16[1]}, [%2], %26 \n"

                        // float32x4_t t7135 = vmlaq_lane_f32(t71, t35, vget_high_f32(coeff1), 1);
                        "vmla.f32   q6, q7, %f25[1]     \n"

                        // r20[3] = vgetq_lane_f32(r0_tm3, 2);
                        // r30[3] = vgetq_lane_f32(r0_tm3, 3);
                        "vst1.f32   {d17[0]}, [%4], %26 \n"
                        "vst1.f32   {d17[1]}, [%6], %26 \n"

                        // vtrn_type: 将两个输入vector的元素通过转置生成一个有两个vector的矩阵
                        // 如：src.val[0] = {1,2,3,4,5,6,7,8}
                        // src.val[1] = {9,10,11,12,13,14,15,16}
                        // dst = vtrn_u8(src.val[0], src.val[1])时，
                        // 则 dst.val[0] = {1,9, 3,11,5,13,7,15}
                        // dst.val[1] = {2,10,4,12,6,14,8,16}

                        // q9: float32x4_t r0_tm4 = vsubq_f32(tmp34a, tmp34b);
                        // q9 => [a, b, c, d]
                        
                        // q2: float32x4_t r0_tm_4_1 = vaddq_f32(tmp56a, tmp56b);
                        // q2 => [e, f, g, h]

                        // q3: float32x4_t r0_tm_4_2 = vsubq_f32(tmp56a, tmp56b);
                        // q3 => [i, j, k, l]

                        // q6: float32x4_t t7135 = vmlaq_lane_f32(t71, t35, vget_high_f32(coeff1), 1);
                        // q6 => [m, n, o, p]

                        // r04[0] = vgetq_lane_f32(r0_tm4, 0);
                        // r14[0] = vgetq_lane_f32(r0_tm4, 1);
                        // r24[0] = vgetq_lane_f32(r0_tm4, 2);
                        // r34[0] = vgetq_lane_f32(r0_tm4, 3);

                        // r04[1] = vgetq_lane_f32(r0_tm_4_1, 0);
                        // r14[1] = vgetq_lane_f32(r0_tm_4_1, 1);
                        // r24[1] = vgetq_lane_f32(r0_tm_4_1, 2);
                        // r34[1] = vgetq_lane_f32(r0_tm_4_1, 3);

                        // r04[2] = vgetq_lane_f32(r0_tm_4_2, 0);
                        // r14[2] = vgetq_lane_f32(r0_tm_4_2, 1);
                        // r24[2] = vgetq_lane_f32(r0_tm_4_2, 2);
                        // r34[2] = vgetq_lane_f32(r0_tm_4_2, 3);

                        // r04[3] = vgetq_lane_f32(t7135, 0);
                        // r14[3] = vgetq_lane_f32(t7135, 1);
                        // r24[3] = vgetq_lane_f32(t7135, 2);
                        // r34[3] = vgetq_lane_f32(t7135, 3);

                        // q9 = [a, e, c, g]
                        // q2 = [b, f, d, h]
                        "vtrn.32    q9, q2              \n"
                        // q3 = [i, m, k, o]
                        // q6 = [j, n, l, p]
                        "vtrn.32    q3, q6              \n"

                        // 多移动了3*4=12个字节
                        "sub        %0, %0, #12         \n"
                        "sub        %2, %2, #12         \n"
                        "sub        %4, %4, #12         \n"
                        "sub        %6, %6, #12         \n"

                        // q9 = [a, e, i, m]
                        // q3 = [c, g, k, o]
                        "vswp       d19, d6             \n"
                        // q2 = [b, f, j, n]
                        // q6 = [d, h, l, p]
                        "vswp       d5, d12             \n"

                        "vst1.f32   {d18-d19}, [%1], %26 \n"
                        "vst1.f32   {d4-d5}, [%3], %26  \n"
                        "vst1.f32   {d6-d7}, [%5], %26  \n"
                        "vst1.f32   {d12-d13}, [%7], %26 \n"

                        /**************************************************************/
                        // loop2
                        // float32x4_t t0_0123 = vld1q_f32(t0);
                        // float32x4_t t0_4567 = vld1q_f32(t0 + 4);
                        "vld1.f32   {d16-d19}, [%8]     \n"
                        // t0 += 8*4 -> t0 += 32*4 = 128
                        "add        %8, %8, #128        \n"
                        // float32x4_t t1_0123 = vld1q_f32(t1);
                        // float32x4_t t1_4567 = vld1q_f32(t1 + 4);
                        "vld1.f32   {d20-d23}, [%9]     \n"
                        "add        %9, %9, #128        \n"
                        // float32x4_t t2_0123 = vld1q_f32(t2);
                        // float32x4_t t2_4567 = vld1q_f32(t2 + 4);
                        "vld1.f32   {d24-d27}, [%10]    \n"
                        "add        %10, %10, #128      \n"
                        // float32x4_t t3_0123 = vld1q_f32(t3);
                        // float32x4_t t3_4567 = vld1q_f32(t3 + 4);
                        "vld1.f32   {d28-d31}, [%11]    \n"
                        "add        %11, %11, #128      \n"

                        // float32x4x2_t t01_00221133 = vtrnq_f32(t0_0123, t1_0123);
                        "vtrn.32    q8, q10             \n"
                        // float32x4x2_t t01_44665577 = vtrnq_f32(t0_4567, t1_4567);
                        "vtrn.32    q9, q11             \n"
                        // float32x4x2_t t23_00221133 = vtrnq_f32(t2_0123, t3_0123);
                        "vtrn.32    q12, q14            \n"
                        // float32x4x2_t t23_44665577 = vtrnq_f32(t2_4567, t3_4567);
                        "vtrn.32    q13, q15            \n"

                        //  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        // q12 = 22  q13 = 66  q14 = 33  q15 = 77
                        // float32x4_t t_00 = vcombine_f32(vget_low_f32(t01_00221133.val[0]), vget_low_f32(t23_00221133.val[0]));
                        // float32x4_t t_11 = vcombine_f32(vget_low_f32(t01_00221133.val[1]), vget_low_f32(t23_00221133.val[1]));
                        // float32x4_t t_22 = vcombine_f32(vget_high_f32(t01_00221133.val[0]), vget_high_f32(t23_00221133.val[0]));
                        // float32x4_t t_33 = vcombine_f32(vget_high_f32(t01_00221133.val[1]), vget_high_f32(t23_00221133.val[1]));
                        // float32x4_t t_44 = vcombine_f32(vget_low_f32(t01_44665577.val[0]), vget_low_f32(t23_44665577.val[0]));
                        // float32x4_t t_55 = vcombine_f32(vget_low_f32(t01_44665577.val[1]), vget_low_f32(t23_44665577.val[1]));
                        // float32x4_t t_66 = vcombine_f32(vget_high_f32(t01_44665577.val[0]), vget_high_f32(t23_44665577.val[0]));
                        // float32x4_t t_77 = vcombine_f32(vget_high_f32(t01_44665577.val[1]), vget_high_f32(t23_44665577.val[1]));
                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"
                        "vswp       d23, d30            \n"

                        // float32x4_t t06 = vsubq_f32(t_00, t_66);
                        "vsub.f32   q2, q8, q13         \n"
                        // float32x4_t t42 = vsubq_f32(t_44, t_22);
                        "vsub.f32   q3, q9, q12         \n"

                        // float32x4_t t26 = vaddq_f32(t_22, t_66);
                        "vadd.f32   q4, q12, q13        \n"
                        // float32x4_t t15 = vaddq_f32(t_11, t_55);
                        "vadd.f32   q5, q10, q11        \n"
                        // float32x4_t t0642 = vmlaq_lane_f32(t06, t42, vget_high_f32(coeff1), 1);
                        "vmla.f32   q2, q3, %f25[1]     \n"
                        // float32x4_t t3c = vmulq_lane_f32(t_33, vget_low_f32(coeff1), 0);
                        "vmul.f32   q7, q14, %e25[0]    \n" 
                        // float32x4_t t4c = vmulq_lane_f32(t_44, vget_high_f32(coeff0), 0);
                        "vmul.f32   q6, q9, %f24[0]     \n" 

                        // float32x4_t tmp12a = vmlsq_lane_f32(t26, t_44, vget_high_f32(coeff1), 0);
                        "vmls.f32   q4, q9, %f25[0]     \n"
                        // float32x4_t tmp12b = vmlsq_lane_f32(t15, t_33, vget_high_f32(coeff1), 0);
                        "vmls.f32   q5, q14, %f25[0]    \n"

                        // r00[0] = vgetq_lane_f32(t0642, 0);
                        // r10[0] = vgetq_lane_f32(t0642, 1);
                        "vst1.f32   {d4[0]}, [%0]!      \n"
                        "vst1.f32   {d4[1]}, [%2]!      \n"

                        "vmov       q3, q7              \n"  //q3 = q7

                        // r20[0] = vgetq_lane_f32(t0642, 2);
                        // r30[0] = vgetq_lane_f32(t0642, 3);
                        "vst1.f32   {d5[0]}, [%4]!      \n"
                        "vst1.f32   {d5[1]}, [%6]!      \n"
                        // float32x4_t tmp34a = vaddq_f32(t_66, t4c);
                        "vadd.f32   q2, q13, q6         \n" 
                        // float32x4_t tmp34b = vmlaq_lane_f32(t3c, t_11, vget_low_f32(coeff0), 1);
                        "vmla.f32   q3, q10, %e24[1]    \n"

                        // float32x4_t r0_tm1 = vaddq_f32(tmp12a, tmp12b);
                        "vadd.f32   q8, q4, q5          \n"
                        // float32x4_t r0_tm2 = vsubq_f32(tmp12a, tmp12b);
                        "vsub.f32   q9, q4, q5          \n"

                        "vmov       q5, q7              \n" //q5 = q7 = t3c
                        
                        // float32x4_t t2c = vaddq_f32(t_22, t4c);
                        "vadd.f32   q6, q12, q6         \n" 
                        // float32x4_t tmp56b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_high_f32(_coeff0), 1);
                        "vmla.f32   q5, q10, %f24[1]    \n"

                        "vmov       q4, q13             \n" //q4 = q13 = t_66

                        // tmp34a = vmlaq_lane_f32(tmp34, t_22, vget_low_f32(coeff0), 0);
                        "vmla.f32   q2, q12, %e24[0]    \n"
                        // tmp34b = vmlaq_lane_f32(tmp34b, t_55, vget_high_f32(coeff0), 1);
                        "vmla.f32   q3, q11, %f24[1]    \n"

                        // r00[1] = vgetq_lane_f32(r0_tm1, 0);
                        // r10[1] = vgetq_lane_f32(r0_tm1, 1);
                        "vst1.f32   {d16[0]}, [%0]!     \n"
                        "vst1.f32   {d16[1]}, [%2]!     \n"

                        // float32x4_t tmp56a = vmlaq_lane_f32(t_66, t2c, vget_low_f32(_coeff1), 1);
                        "vmla.f32   q4, q6, %e25[1]     \n"

                        // r20[1] = vgetq_lane_f32(r0_tm1, 2);
                        // r30[1] = vgetq_lane_f32(r0_tm1, 3);
                        "vst1.f32   {d17[0]}, [%4]!     \n"
                        "vst1.f32   {d17[1]}, [%6]!     \n"

                        // tmp56b = vmlaq_lane_f32(tmp56b, t_55, vget_low_f32(coeff0), 1);
                        "vmla.f32   q5, q11, %e24[1]    \n"

                        // r00[2] = vgetq_lane_f32(r0_tm2, 0);
                        // r10[2] = vgetq_lane_f32(r0_tm2, 1);
                        "vst1.f32   {d18[0]}, [%0]!     \n"
                        "vst1.f32   {d18[1]}, [%2]!     \n"

                        // float32x4_t r0_tm3 = vaddq_f32(tmp34a, tmp34b);
                        "vadd.f32   q8, q2, q3          \n"

                        // r20[2] = vgetq_lane_f32(r0_tm2, 2);
                        // r30[2] = vgetq_lane_f32(r0_tm2, 3);
                        "vst1.f32   {d19[0]}, [%4]!     \n"
                        "vst1.f32   {d19[1]}, [%6]!     \n"

                        // float32x4_t r0_tm4 = vsubq_f32(tmp34a, tmp34b);
                        "vsub.f32   q9, q2, q3          \n"

                        // float32x4_t t71 = vsubq_f32(t_77, t_11);
                        "vsub.f32   q6, q15, q10        \n"
                        // float32x4_t t35 = vsubq_f32(t_33, t_55);
                        "vsub.f32   q7, q14, q11        \n"

                        // float32x4_t r0_tm_4_1 = vaddq_f32(tmp56a, tmp56b);
                        "vadd.f32   q2, q4, q5          \n"
                        // float32x4_t r0_tm_4_2 = vsubq_f32(tmp56a, tmp56b);
                        "vsub.f32   q3, q4, q5          \n"

                        // r00[3] = vgetq_lane_f32(r0_tm3, 0);
                        // r10[3] = vgetq_lane_f32(r0_tm3, 1);
                        "vst1.f32   {d16[0]}, [%0], %26 \n"
                        "vst1.f32   {d16[1]}, [%2], %26 \n"

                        // float32x4_t t7135 = vmlaq_lane_f32(t71, t35, vget_high_f32(coeff1), 1);
                        "vmla.f32   q6, q7, %f25[1]     \n"

                        // r20[3] = vgetq_lane_f32(r0_tm3, 2);
                        // r30[3] = vgetq_lane_f32(r0_tm3, 3);
                        "vst1.f32   {d17[0]}, [%4], %26 \n"
                        "vst1.f32   {d17[1]}, [%6], %26 \n"

                        // vtrn_type: 将两个输入vector的元素通过转置生成一个有两个vector的矩阵
                        // 如：src.val[0] = {1,2,3,4,5,6,7,8}
                        // src.val[1] = {9,10,11,12,13,14,15,16}
                        // dst = vtrn_u8(src.val[0], src.val[1])时，
                        // 则 dst.val[0] = {1,9, 3,11,5,13,7,15}
                        // dst.val[1] = {2,10,4,12,6,14,8,16}

                        // q9: float32x4_t r0_tm4 = vsubq_f32(tmp34a, tmp34b);
                        // q9 => [a, b, c, d]
                        
                        // q2: float32x4_t r0_tm_4_1 = vaddq_f32(tmp56a, tmp56b);
                        // q2 => [e, f, g, h]

                        // q3: float32x4_t r0_tm_4_2 = vsubq_f32(tmp56a, tmp56b);
                        // q3 => [i, j, k, l]

                        // q6: float32x4_t t7135 = vmlaq_lane_f32(t71, t35, vget_high_f32(coeff1), 1);
                        // q6 => [m, n, o, p]

                        // r04[0] = vgetq_lane_f32(r0_tm4, 0);
                        // r14[0] = vgetq_lane_f32(r0_tm4, 1);
                        // r24[0] = vgetq_lane_f32(r0_tm4, 2);
                        // r34[0] = vgetq_lane_f32(r0_tm4, 3);

                        // r04[1] = vgetq_lane_f32(r0_tm_4_1, 0);
                        // r14[1] = vgetq_lane_f32(r0_tm_4_1, 1);
                        // r24[1] = vgetq_lane_f32(r0_tm_4_1, 2);
                        // r34[1] = vgetq_lane_f32(r0_tm_4_1, 3);

                        // r04[2] = vgetq_lane_f32(r0_tm_4_2, 0);
                        // r14[2] = vgetq_lane_f32(r0_tm_4_2, 1);
                        // r24[2] = vgetq_lane_f32(r0_tm_4_2, 2);
                        // r34[2] = vgetq_lane_f32(r0_tm_4_2, 3);

                        // r04[3] = vgetq_lane_f32(t7135, 0);
                        // r14[3] = vgetq_lane_f32(t7135, 1);
                        // r24[3] = vgetq_lane_f32(t7135, 2);
                        // r34[3] = vgetq_lane_f32(t7135, 3);

                        // q9 = [a, e, c, g]
                        // q2 = [b, f, d, h]
                        "vtrn.32    q9, q2              \n"
                        // q3 = [i, m, k, o]
                        // q6 = [j, n, l, p]
                        "vtrn.32    q3, q6              \n"

                        // 多移动了3*4=12个字节
                        "sub        %0, %0, #12         \n"
                        "sub        %2, %2, #12         \n"
                        "sub        %4, %4, #12         \n"
                        "sub        %6, %6, #12         \n"

                        // q9 = [a, e, i, m]
                        // q3 = [c, g, k, o]
                        "vswp       d19, d6             \n"
                        // q2 = [b, f, j, n]
                        // q6 = [d, h, l, p]
                        "vswp       d5, d12             \n"

                        //注意越界问题
                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "vst1.f32   {d4-d5}, [%3]       \n"
                        "vst1.f32   {d6-d7}, [%5]       \n"
                        "vst1.f32   {d12-d13}, [%7]     \n"

                        : "=r"(r00), // %0
                        "=r"(r04), // %1
                        "=r"(r10), // %2
                        "=r"(r14), // %3
                        "=r"(r20), // %4
                        "=r"(r24), // %5
                        "=r"(r30), // %6
                        "=r"(r34), // %7
                        "=r"(t0),       // %8
                        "=r"(t1),       // %9
                        "=r"(t2),       // %10
                        "=r"(t3)        // %11
                        : "0"(r00),
                        "1"(r04),
                        "2"(r10),
                        "3"(r14),
                        "4"(r20),
                        "5"(r24),
                        "6"(r30),
                        "7"(r34),
                        "8"(t0),
                        "9"(t1),
                        "10"(t2),
                        "11"(t3),
                        "w"(coeff0), // %24
                        "w"(coeff1), // %25
                        "r"(step)     // %26
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif

                   
#else
                    float *r0 = padptr + i * 6 * PadWidth + j * 6;

                    // Bd_{c,b}
                    for(int m = 0; m < 8; m++){

                        tmpV[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25f;
                        tmpV[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25f;

                        float t1 = (r0[2] + r0[6] - r0[4] * 4.25f);
                        float t2 = (r0[1] + r0[5] - r0[3] * 4.25f);

                        tmpV[1][m] = t1 + t2;
                        tmpV[2][m] = t1 - t2;

                        float t3 = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                        float t4 = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);
                        tmpV[3][m] = t3 + t4;
                        tmpV[4][m] = t3 - t4;

                        float t5 = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                        float t6 = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

                        tmpV[5][m] = t5 + t6;
                        tmpV[6][m] = t5 - t6;

                        r0 += PadWidth;
                    }

                    //Bd_{c,b}B^T
                    float *r00 = srcptr + (i * w_tm / 8 + j) * src_tm_w;
                    float *r04 = srcptr + (i * w_tm /8 + j + tiles) * src_tm_w;

                    for(int m = 0; m < 8; m++){
                        float* tmpVPtr = tmpV[m];
                        r00[0] = tmpVPtr[0] - tmpVPtr[6] + (tmpVPtr[4] - tmpVPtr[2]) * 5.25f;
                        r04[3] = tmpVPtr[7] - tmpVPtr[1] + (tmpVPtr[3] - tmpVPtr[5]) * 5.25f;
                        
                        float t1 =  (tmpVPtr[2] + tmpVPtr[6] - tmpVPtr[4] * 4.25f);
                        float t2 =  (tmpVPtr[1] - tmpVPtr[3] * 4.25f + tmpVPtr[5]);
                        r00[1] = t1 + t2;
                        r00[2] = t1 - t2;

                        float t3 = (tmpVPtr[6] + tmpVPtr[2] * 0.25f - tmpVPtr[4] * 1.25);
                        float t4 = (tmpVPtr[1] * 0.5f - tmpVPtr[3] * 2.5f + tmpVPtr[5] * 2.f);
                        r00[3] = t3 + t4;
                        r04[0] = t3 - t4;

                        float t5 = (tmpVPtr[6] + (tmpVPtr[2] - tmpVPtr[4] * 1.25f) * 4.f);
                        float t6 = (tmpVPtr[1] * 2.f - tmpVPtr[3] * 2.5f + tmpVPtr[5] * 0.5f);

                        r04[1] = t5 + t6;
                        r04[2] = t5 - t6;

                        r00 += 2 * tiles * src_tm_w;
                        r04 += 2 * tiles * src_tm_w;

                    }
#endif

                }
            }
        }

        //delete [] srcPadding;

        //Mk,b = \sum_{c}U_{k,c}V_{c,b}
        //k表示outChannel，b表示tile序号
        const int dst_tm_h = src_tm_h;
        const int dst_tm_w = src_tm_w;
        const int dst_tm_size = dst_tm_h * dst_tm_w;
        float *dest_tm = new float[outChannel * dst_tm_h * dst_tm_w];
        const int nnOutChannel = outChannel >> 2;
        const int remainOutChannel = nnOutChannel << 2;
        const int kernelSize = kHeight * kWidth;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = 0; cc < nnOutChannel; cc++){
            int c = cc * 4;
            float *dest0 = dest_tm + c * dst_tm_size;
            float *dest1 = dest_tm + (c + 1) * dst_tm_size;
            float *dest2 = dest_tm + (c + 2) * dst_tm_size;
            float *dest3 = dest_tm + (c + 3) * dst_tm_size;



            const float *ktm = kernel + cc * kernelSize;
            int q = 0;
            
            for(; q + 1 < inChannel; q += 2){
                const float* r0 = src_tm + q * src_tm_size;
                const float* r1 = src_tm + (q + 1) * src_tm_size;
                
                float* destptr0 = dest0;
                float *destptr1 = dest1;
                float *destptr2 = dest2;
                float *destptr3 = dest3;

#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    // for(int r = 0; r < 16; r++) => r0 = 16
                    "mov        r0, #16                 \n"

                    "0:                                 \n"
                    // q0 = k00 & q1 = k01
                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d0-d3}, [%6]!          \n"
                    // q2 = k10, q3 = k11
                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d4-d7}, [%6]!          \n"
                    // q4 = k20, q5 = q21
                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d8-d11}, [%6]!         \n"
                    // q6 = k30, q7 = k31
                    "pld        [%6, #256]              \n"
                    "vld1.f32   {d12-d15}, [%6]!        \n"

                    // for(int t = 0; t < tiles; t++)
                    // r1 = tiles >> 2
                    "lsr        r1, %14, #2             \n"
                    "cmp        r1, #0                  \n"
                    "beq        2f                      \n"

                    // 开始tiles的循环

                    // q12 = r0
                    "pld        [%4, #128]              \n"
                    "vld1.f32   {d24-d25}, [%4]!        \n"

                    "1:                                 \n"
                    // q8 = destptr0[m]
                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0]         \n"
                    // destptr0[m] += r0[m] * ktm[m];
                    "vmla.f32   q8, q12, q0             \n"
                    // q9 = destptr1[m]
                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1]         \n"
                    // destptr1[m] += r0[m] * ktm[m + 8];
                    "vmla.f32   q9, q12, q2             \n"
                    // q10 = destptr2[m]
                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2]         \n"
                    // destptr2[m] += r0[m] * ktm[m + 16];
                    "vmla.f32   q10, q12, q4            \n"
                    // q11 = destptr3[m]
                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3]         \n"
                    // destptr3[m] += r0[m] * ktm[m + 24];
                    "vmla.f32   q11, q12, q6            \n"

                    // q13 = r1
                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5]!        \n"
                    // destptr0[m] += r1[m] * ktm[m + 4];
                    "vmla.f32   q8, q13, q1             \n"
                    // destptr1[m] += r1[m] * ktm[m + 12];
                    "vmla.f32   q9, q13, q3             \n"
                    // q12 = r0
                    "pld        [%4, #128]              \n" 
                    "vld1.f32   {d24-d25}, [%4]!        \n" 
                    // destptr2[m] += r1[m] * ktm[m + 20];
                    "vmla.f32   q10, q13, q5            \n"
                    // destptr3[m] += r1[m] * ktm[m + 28];
                    "vmla.f32   q11, q13, q7            \n"
                    
                    // 
                    "vst1.f32   {d16-d17}, [%0]!        \n"
                    "vst1.f32   {d18-d19}, [%1]!        \n"
                    "vst1.f32   {d20-d21}, [%2]!        \n"
                    "vst1.f32   {d22-d23}, [%3]!        \n"

                    
                    // q8 = destptr0[m]
                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0]         \n"
                    // destptr0[m] += r0[m] * ktm[m];
                    "vmla.f32   q8, q12, q0             \n"
                    // q9 = destptr1[m]
                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1]         \n"
                    // destptr1[m] += r0[m] * ktm[m + 8];
                    "vmla.f32   q9, q12, q2             \n"
                    // q10 = destptr2[m]
                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2]         \n"
                    // destptr2[m] += r0[m] * ktm[m + 16];
                    "vmla.f32   q10, q12, q4            \n"
                    // q11 = destptr3[m]
                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3]         \n"
                    // destptr3[m] += r0[m] * ktm[m + 24];
                    "vmla.f32   q11, q12, q6            \n"

                    // q13 = r1
                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5]!        \n"
                    // destptr0[m] += r1[m] * ktm[m + 4];
                    "vmla.f32   q8, q13, q1             \n"
                    // destptr1[m] += r1[m] * ktm[m + 12];
                    "vmla.f32   q9, q13, q3             \n"
                    // q12 = r0
                    "pld        [%4, #128]              \n" 
                    "vld1.f32   {d24-d25}, [%4]!        \n" 
                    // destptr2[m] += r1[m] * ktm[m + 20];
                    "vmla.f32   q10, q13, q5            \n"
                    // destptr3[m] += r1[m] * ktm[m + 28];
                    "vmla.f32   q11, q13, q7            \n"
                    
                    // 
                    "vst1.f32   {d16-d17}, [%0]!        \n"
                    "vst1.f32   {d18-d19}, [%1]!        \n"
                    "vst1.f32   {d20-d21}, [%2]!        \n"
                    "vst1.f32   {d22-d23}, [%3]!        \n"

                    // q8 = destptr0[m]
                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0]         \n"
                    // destptr0[m] += r0[m] * ktm[m];
                    "vmla.f32   q8, q12, q0             \n"
                    // q9 = destptr1[m]
                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1]         \n"
                    // destptr1[m] += r0[m] * ktm[m + 8];
                    "vmla.f32   q9, q12, q2             \n"
                    // q10 = destptr2[m]
                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2]         \n"
                    // destptr2[m] += r0[m] * ktm[m + 16];
                    "vmla.f32   q10, q12, q4            \n"
                    // q11 = destptr3[m]
                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3]         \n"
                    // destptr3[m] += r0[m] * ktm[m + 24];
                    "vmla.f32   q11, q12, q6            \n"

                    // q13 = r1
                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5]!        \n"
                    // destptr0[m] += r1[m] * ktm[m + 4];
                    "vmla.f32   q8, q13, q1             \n"
                    // destptr1[m] += r1[m] * ktm[m + 12];
                    "vmla.f32   q9, q13, q3             \n"
                    // q12 = r0
                    "pld        [%4, #128]              \n" 
                    "vld1.f32   {d24-d25}, [%4]!        \n" 
                    // destptr2[m] += r1[m] * ktm[m + 20];
                    "vmla.f32   q10, q13, q5            \n"
                    // destptr3[m] += r1[m] * ktm[m + 28];
                    "vmla.f32   q11, q13, q7            \n"
                    
                    // 
                    "vst1.f32   {d16-d17}, [%0]!        \n"
                    "vst1.f32   {d18-d19}, [%1]!        \n"
                    "vst1.f32   {d20-d21}, [%2]!        \n"
                    "vst1.f32   {d22-d23}, [%3]!        \n"

                    // q8 = destptr0[m]
                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0]         \n"
                    // destptr0[m] += r0[m] * ktm[m];
                    "vmla.f32   q8, q12, q0             \n"
                    // q9 = destptr1[m]
                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1]         \n"
                    // destptr1[m] += r0[m] * ktm[m + 8];
                    "vmla.f32   q9, q12, q2             \n"
                    // q10 = destptr2[m]
                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2]         \n"
                    // destptr2[m] += r0[m] * ktm[m + 16];
                    "vmla.f32   q10, q12, q4            \n"
                    // q11 = destptr3[m]
                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3]         \n"
                    // destptr3[m] += r0[m] * ktm[m + 24];
                    "vmla.f32   q11, q12, q6            \n"

                    // q13 = r1
                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5]!        \n"
                    // destptr0[m] += r1[m] * ktm[m + 4];
                    "vmla.f32   q8, q13, q1             \n"
                    // destptr1[m] += r1[m] * ktm[m + 12];
                    "vmla.f32   q9, q13, q3             \n"
                    // q12 = r0
                    "pld        [%4, #128]              \n" 
                    "vld1.f32   {d24-d25}, [%4]!        \n" 
                    // destptr2[m] += r1[m] * ktm[m + 20];
                    "vmla.f32   q10, q13, q5            \n"
                    // destptr3[m] += r1[m] * ktm[m + 28];
                    "vmla.f32   q11, q13, q7            \n"
                    
                    // 
                    "vst1.f32   {d16-d17}, [%0]!        \n"
                    "vst1.f32   {d18-d19}, [%1]!        \n"
                    "vst1.f32   {d20-d21}, [%2]!        \n"
                    "vst1.f32   {d22-d23}, [%3]!        \n"
                    

                    "subs       r1, #1                  \n"
                    "bne        1b                      \n"
                    "sub        %4, %4, #16             \n"

                    // tiles循环结束

                    "2:                                 \n"
                    // r1 = remain = tiles & 3
                    "and        r1, %14, #3             \n"
                    "cmp        r1, #0                  \n"
                    "beq        4f                      \n"

                    "3:                                 \n"

                    // q12 = r0
                    "pld        [%4, #128]              \n" 
                    "vld1.f32   {d24-d25}, [%4]!        \n" 

                    // q8 = destptr0[m]
                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0]         \n"
                    // destptr0[m] += r0[m] * ktm[m];
                    "vmla.f32   q8, q12, q0             \n"
                    // q9 = destptr1[m]
                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1]         \n"
                    // destptr1[m] += r0[m] * ktm[m + 8];
                    "vmla.f32   q9, q12, q2             \n"
                    // q10 = destptr2[m]
                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2]         \n"
                    // destptr2[m] += r0[m] * ktm[m + 16];
                    "vmla.f32   q10, q12, q4            \n"
                    // q11 = destptr3[m]
                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3]         \n"
                    // destptr3[m] += r0[m] * ktm[m + 24];
                    "vmla.f32   q11, q12, q6            \n"

                    // q13 = r1
                    "pld        [%5, #128]              \n"
                    "vld1.f32   {d26-d27}, [%5]!        \n"
                    // destptr0[m] += r1[m] * ktm[m + 4];
                    "vmla.f32   q8, q13, q1             \n"
                    // destptr1[m] += r1[m] * ktm[m + 12];
                    "vmla.f32   q9, q13, q3             \n"
                    // destptr2[m] += r1[m] * ktm[m + 20];
                    "vmla.f32   q10, q13, q5            \n"
                    // destptr3[m] += r1[m] * ktm[m + 28];
                    "vmla.f32   q11, q13, q7            \n"
                    
                    // 
                    "vst1.f32   {d16-d17}, [%0]!        \n"
                    "vst1.f32   {d18-d19}, [%1]!        \n"
                    "vst1.f32   {d20-d21}, [%2]!        \n"
                    "vst1.f32   {d22-d23}, [%3]!        \n"

                    "subs       r1, #1                  \n"
                    "bne        3b                      \n"

                    "4:                                 \n"
                    "subs       r0, #1                  \n"
                    "bne        0b                      \n"


                    : "=r"(destptr0), // %0
                    "=r"(destptr1), // %1
                    "=r"(destptr2), // %2
                    "=r"(destptr3), // %3
                    "=r"(r0),         // %4
                    "=r"(r1),         // %5
                    "=r"(ktm)         // %6
                    : "0"(destptr0),
                    "1"(destptr1),
                    "2"(destptr2),
                    "3"(destptr3),
                    "4"(r0),
                    "5"(r1),
                    "6"(ktm),
                    "r"(tiles) // %14
                    : "cc", "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13"
                );
#endif

#else

                for(int r = 0; r < 16; r++){
                    for(int t = 0; t < tiles; t++){
                        for(int m = 0; m < 4; m++){
                            destptr0[m] += r0[m] * ktm[m];
                            destptr0[m] += r1[m] * ktm[m + 4];
                            destptr1[m] += r0[m] * ktm[m + 8];
                            destptr1[m] += r1[m] * ktm[m + 12];
                            destptr2[m] += r0[m] * ktm[m + 16];
                            destptr2[m] += r1[m] * ktm[m + 20];
                            destptr3[m] += r0[m] * ktm[m + 24];
                            destptr3[m] += r1[m] * ktm[m + 28];  
                        }

                        r0 += 4;
                        r1 += 4;
                        destptr0 += 4;
                        destptr1 += 4;
                        destptr2 += 4;
                        destptr3 += 4;
                    }

                    ktm += 32;
                }
#endif
            }

            for(; q < inChannel; q++){
                const float *r0 = src_tm + q * src_tm_size;
                float* destptr0 = dest0;
                float *destptr1 = dest1;
                float *destptr2 = dest2;
                float *destptr3 = dest3;

#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    // for(int r = 0; r < 16; r++)
                    "mov        r0, #16                 \n"
                    "0:                                 \n"
                    // q0 = k00 & q1 = k01
                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d0-d3}, [%5]!          \n"
                    // q2 = k10, q3 = k11
                    "pld        [%5, #256]              \n"
                    "vld1.f32   {d4-d7}, [%5]!          \n"

                    // tiles 循环, r1 = tiles
                    "mov        r1, %12                 \n"
                    "cmp        r1, #0                  \n"
                    "beq        2f                      \n"

                    "1:                                 \n"

                    // q12 = r0
                    "pld        [%4, #128]              \n" 
                    "vld1.f32   {d24-d25}, [%4]!        \n" 

                    // q8 = destptr0[m]
                    "pld        [%0, #128]              \n"
                    "vld1.f32   {d16-d17}, [%0]         \n"
                    // destptr0[m] += r0[m] * ktm[m];
                    "vmla.f32   q8, q12, q0             \n"
                    // q9 = destptr1[m]
                    "pld        [%1, #128]              \n"
                    "vld1.f32   {d18-d19}, [%1]         \n"
                    // destptr1[m] += r0[m] * ktm[m + 8];
                    "vmla.f32   q9, q12, q1             \n"
                    // q10 = destptr2[m]
                    "pld        [%2, #128]              \n"
                    "vld1.f32   {d20-d21}, [%2]         \n"
                    // destptr2[m] += r0[m] * ktm[m + 16];
                    "vmla.f32   q10, q12, q2            \n"
                    // q11 = destptr3[m]
                    "pld        [%3, #128]              \n"
                    "vld1.f32   {d22-d23}, [%3]         \n"
                    // destptr3[m] += r0[m] * ktm[m + 24];
                    "vmla.f32   q11, q12, q3            \n"

                    "vst1.f32   {d16-d17}, [%0]!        \n"
                    "vst1.f32   {d18-d19}, [%1]!        \n"
                    "vst1.f32   {d20-d21}, [%2]!        \n"
                    "vst1.f32   {d22-d23}, [%3]!        \n"

                    "subs       r1, #1                  \n"
                    "bne        1b                      \n"

                    "2:                                 \n"
                    "subs       r0, #1                  \n"
                    "bne        0b                      \n"

                    : "=r"(destptr0), // %0
                    "=r"(destptr1), // %1
                    "=r"(destptr2), // %2
                    "=r"(destptr3), // %3
                    "=r"(r0),         // %4
                    "=r"(ktm)         // %5
                    : "0"(destptr0),
                    "1"(destptr1),
                    "2"(destptr2),
                    "3"(destptr3),
                    "4"(r0),
                    "5"(ktm),
                    "r"(tiles) // %12
                    : "cc", "memory", "r0", "r1", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13"
                );
#endif

#else
                for(int r = 0; r < 16; r++){
                    for(int t = 0; t < tiles; t++){
                        for(int m = 0; m < 4; m++){
                            destptr0[m] += r0[m] * ktm[m];
                            destptr1[m] += r0[m] * ktm[m + 4];
                            destptr2[m] += r0[m] * ktm[m + 8];
                            destptr3[m] += r0[m] * ktm[m + 12];
                        }

                        r0 += 4;
                        destptr0 += 4;
                        destptr1 += 4;
                        destptr2 += 4;
                        destptr3 += 4;
                    }

                    ktm += 16;
                }
#endif

            }
        }

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = remainOutChannel; cc < outChannel; cc++){
            int c = cc;
            float *dest0 = dest_tm + c * dst_tm_size;
            const float *ktm = kernel + nnOutChannel * kernelSize + 8 * 8 * inChannel * (c - remainOutChannel);

            int q = 0;

            for(; q < inChannel; q++){
                const float* r0 = src_tm + q * src_tm_size;
                float* destptr0 = dest0;
                for(int r = 0; r < 16; r++){
#if USE_NEON
                    float32x4_t k00 = vld1q_f32(ktm);
#endif // __ARM_NEON

                    for(int i = 0; i < tiles; i++){
#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else

                        asm volatile(
                            "pld        [%1, #128]              \n"
                            "vld1.f32   {d18-d19}, [%1]!        \n" // q9 = _r0

                            "pld        [%0, #128]              \n"
                            "vld1.f32   {d16-d17}, [%0]         \n" // q8 = destptr0

                            "vmla.f32   q8, q9, %q4             \n"

                            "vst1.f32   {d16-d17}, [%0]!        \n"
                            : "=r"(destptr0), // %0
                            "=r"(r0)          // %1
                            : "0"(destptr0),
                            "1"(r0),
                            "w"(k00) // %4
                            : "cc", "memory", "q8", "q9");
#endif

#else
                        for(int m = 0; m < 4; m++){
                            destptr0[m] += r0[m] * ktm[m];
                        }
                        r0 += 4;
                        destptr0 += 4;
#endif
                    }

                    ktm += 4;
                }
            }
        }

        //delete [] src_tm;

// Yk,b=A^TMk,bA
// AT=
// ⎡1  1  1   1    1    1      1    0⎤
// ⎢                                 ⎥
// ⎢0  1  -1  2   -2   1/2   -1/2   0⎥
// ⎢                                 ⎥
// ⎢0  1  1   4    4   1/4    1/4   0⎥
// ⎢                                 ⎥
// ⎢0  1  -1  8   -8   1/8   -1/8   0⎥
// ⎢                                 ⎥
// ⎢0  1  1   16  16   1/16  1/16   0⎥
// ⎢                                 ⎥
// ⎣0  1  -1  32  -32  1/32  -1/32  1⎦

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)


#if USE_NEON
        const float coeff[4] = {4.f, 8.f, 16.f, 32.f};
        float32x4_t _coeff = vld1q_f32(coeff);
#endif

        float *dest_tm2 = new float[outW * outH * outChannel];
        const int dst_tm_size2 = outW * outH;
        

        const int outSize = outHeight * outWidth;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = 0; cc < outChannel; cc++){
            float *destptr = dest_tm + cc * dst_tm_size;
            float *outptr = dest_tm2 + cc * dst_tm_size2;

            float tmpA[6][8];

            for(int i = 0; i < outH / 6; i++){
                for(int j = 0; j < outW / 6; j++){

#if USE_NEON
                    const float* destptr00 = destptr + (i * w_tm / 8 + j) * dst_tm_w;
                    const float* destptr04 = destptr + (i * w_tm / 8 + j + tiles) * dst_tm_w;
                    const float* destptr10 = destptr + (i * w_tm / 8 + j + tiles * 2) * dst_tm_w;
                    const float* destptr14 = destptr + (i * w_tm / 8 + j + tiles * 3) * dst_tm_w;
                    const float* destptr20 = destptr + (i * w_tm / 8 + j + tiles * 4) * dst_tm_w;
                    const float* destptr24 = destptr + (i * w_tm / 8 + j + tiles * 5) * dst_tm_w;
                    const float* destptr30 = destptr + (i * w_tm / 8 + j + tiles * 6) * dst_tm_w;
                    const float* destptr34 = destptr + (i * w_tm / 8 + j + tiles * 7) * dst_tm_w;

#if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
                    for(int m = 0; m + 3 < 8; m += 4){
                        float32x4_t _output0_tm0_0123 = vld1q_f32(destptr00);
                        float32x4_t _output0_tm0_4567 = vld1q_f32(destptr04);
                        float32x4_t _output0_tm1_0123 = vld1q_f32(destptr10);
                        float32x4_t _output0_tm1_4567 = vld1q_f32(destptr14);
                        float32x4_t _output0_tm2_0123 = vld1q_f32(destptr20);
                        float32x4_t _output0_tm2_4567 = vld1q_f32(destptr24);
                        float32x4_t _output0_tm3_0123 = vld1q_f32(destptr30);
                        float32x4_t _output0_tm3_4567 = vld1q_f32(destptr34);

                        float32x4x2_t _output0_tm01_00221133 = vtrnq_f32(_output0_tm0_0123, _output0_tm1_0123);
                        float32x4x2_t _output0_tm01_44665577 = vtrnq_f32(_output0_tm0_4567, _output0_tm1_4567);
                        float32x4x2_t _output0_tm23_00221133 = vtrnq_f32(_output0_tm2_0123, _output0_tm3_0123);
                        float32x4x2_t _output0_tm23_44665577 = vtrnq_f32(_output0_tm2_4567, _output0_tm3_4567);

                        float32x4_t _output0_tm_00 = vcombine_f32(vget_low_f32(_output0_tm01_00221133.val[0]), vget_low_f32(_output0_tm23_00221133.val[0]));
                        float32x4_t _output0_tm_11 = vcombine_f32(vget_low_f32(_output0_tm01_00221133.val[1]), vget_low_f32(_output0_tm23_00221133.val[1]));
                        float32x4_t _output0_tm_22 = vcombine_f32(vget_high_f32(_output0_tm01_00221133.val[0]), vget_high_f32(_output0_tm23_00221133.val[0]));
                        float32x4_t _output0_tm_33 = vcombine_f32(vget_high_f32(_output0_tm01_00221133.val[1]), vget_high_f32(_output0_tm23_00221133.val[1]));
                        float32x4_t _output0_tm_44 = vcombine_f32(vget_low_f32(_output0_tm01_44665577.val[0]), vget_low_f32(_output0_tm23_44665577.val[0]));
                        float32x4_t _output0_tm_55 = vcombine_f32(vget_low_f32(_output0_tm01_44665577.val[1]), vget_low_f32(_output0_tm23_44665577.val[1]));
                        float32x4_t _output0_tm_66 = vcombine_f32(vget_high_f32(_output0_tm01_44665577.val[0]), vget_high_f32(_output0_tm23_44665577.val[0]));
                        float32x4_t _output0_tm_77 = vcombine_f32(vget_high_f32(_output0_tm01_44665577.val[1]), vget_high_f32(_output0_tm23_44665577.val[1]));

                        // float t1 = destptr0[1] + destptr0[2];
                        float32x4_t _tmp024a = vaddq_f32(_output0_tm_11, _output0_tm_22);
                        // float t2 = destptr0[1] - destptr0[2];
                        float32x4_t _tmp135a = vsubq_f32(_output0_tm_11, _output0_tm_22);

                        // float t3 = destptr0[3] + destptr4[0];
                        float32x4_t _tmp024b = vaddq_f32(_output0_tm_33, _output0_tm_44);
                        // float t4 = destptr0[3] - destptr4[0];
                        float32x4_t _tmp135b = vsubq_f32(_output0_tm_33, _output0_tm_44);

                        // float t5 = destptr4[1] + destptr4[2];
                        float32x4_t _tmp024c = vaddq_f32(_output0_tm_55, _output0_tm_66);
                        // float t6 = destptr4[1] - destptr4[2];
                        float32x4_t _tmp135c = vsubq_f32(_output0_tm_55, _output0_tm_66);

                        // tmpA[0][m] = destptr0[0] + t1 + t3 + t5 * 32;
                        // destptr0[0] + t1
                        float32x4_t _tmp0 = vaddq_f32(_output0_tm_00, _tmp024a);
                        // + t5 * 32
                        _tmp0 = vmlaq_lane_f32(_tmp0, _tmp024c, vget_high_f32(_coeff), 1);
                        // + t3
                        _tmp0 = vaddq_f32(_tmp0, _tmp024b);

                        // tmpA[2][m] = t1 + t3 * 4 + t5 * 8;
                        // t1 + t3 * 4
                        float32x4_t _tmp2 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        // + t5*8
                        _tmp2 = vmlaq_lane_f32(_tmp2, _tmp024c, vget_low_f32(_coeff), 1);

                        // tmpA[4][m] = t1 + t3 * 16 + t5 + t5;
                        // t1 + t3*16
                        float32x4_t _tmp4 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        // + t5
                        _tmp4 = vaddq_f32(_tmp4, _tmp024c);
                        // +t5
                        _tmp4 = vaddq_f32(_tmp4, _tmp024c);

                        vst1q_f32(&tmpA[0][m], _tmp0);
                        vst1q_f32(&tmpA[2][m], _tmp2);
                        vst1q_f32(&tmpA[4][m], _tmp4);

                        // tmpA[1][m] = t2 + t4 + t4 + t6 * 16;
                        // t2 + t6 * 16
                        float32x4_t _tmp1 = vmlaq_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        // + t4
                        _tmp1 = vaddq_f32(_tmp1, _tmp135b);
                        // + t4
                        _tmp1 = vaddq_f32(_tmp1, _tmp135b);

                        // tmpA[3][m] = t2 + t4 * 8 + t6 * 4;
                        // t2 + t4 * 8
                        float32x4_t _tmp3 = vmlaq_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        // + t6 * 4
                        _tmp3 = vmlaq_lane_f32(_tmp3, _tmp135c, vget_low_f32(_coeff), 0);

                        // tmpA[5][m] = destptr4[3] + t2 + t4 * 32 + t6;
                        // destptr4[3] + t2
                        float32x4_t _tmp5 = vaddq_f32(_output0_tm_77, _tmp135a);
                        // + t4 * 32
                        _tmp5 = vmlaq_lane_f32(_tmp5, _tmp135b, vget_high_f32(_coeff), 1);
                        // + t6
                        _tmp5 = vaddq_f32(_tmp5, _tmp135c);
                        vst1q_f32(&tmpA[1][m], _tmp1);
                        vst1q_f32(&tmpA[3][m], _tmp3);
                        vst1q_f32(&tmpA[5][m], _tmp5);

                        destptr00 += dst_tm_w * 2 * 4 * tiles;
                        destptr04 += dst_tm_w * 2 * 4 * tiles;
                        destptr10 += dst_tm_w * 2 * 4 * tiles;
                        destptr14 += dst_tm_w * 2 * 4 * tiles;
                        destptr20 += dst_tm_w * 2 * 4 * tiles;
                        destptr24 += dst_tm_w * 2 * 4 * tiles;
                        destptr30 += dst_tm_w * 2 * 4 * tiles;
                        destptr34 += dst_tm_w * 2 * 4 * tiles;
                    }

                    const float* t0 = tmpA[0];
                    const float* t1 = tmpA[1];

                    float* output0 = outptr + (i * 6) * outW + j * 6;
                    float* output1 = output0 + outW;

                    for (int m = 0; m + 1 < 6; m += 2)
                    {

                        float32x4_t _t0_0123 = vld1q_f32(t0);
                        float32x4_t _t0_4567 = vld1q_f32(t0 + 4);
                        float32x4_t _t1_0123 = vld1q_f32(t1);
                        float32x4_t _t1_4567 = vld1q_f32(t1 + 4);

                        float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                        float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);

                        float32x2_t _t_00 = vget_low_f32(_t01_00221133.val[0]);
                        float32x2_t _t_11 = vget_low_f32(_t01_00221133.val[1]);
                        float32x2_t _t_22 = vget_high_f32(_t01_00221133.val[0]);
                        float32x2_t _t_33 = vget_high_f32(_t01_00221133.val[1]);
                        float32x2_t _t_44 = vget_low_f32(_t01_44665577.val[0]);
                        float32x2_t _t_55 = vget_low_f32(_t01_44665577.val[1]);
                        float32x2_t _t_66 = vget_high_f32(_t01_44665577.val[0]);
                        float32x2_t _t_77 = vget_high_f32(_t01_44665577.val[1]);

                        // float t1 = tmp0[1] + tmp0[2];
                        float32x2_t _tmp024a = vadd_f32(_t_11, _t_22);
                        // float t2 = tmp0[1] - tmp0[2];
                        float32x2_t _tmp135a = vsub_f32(_t_11, _t_22);

                        // float t3 = destptr0[3] + destptr4[0];
                        float32x2_t _tmp024b = vadd_f32(_t_33, _t_44);
                        // float t4 = destptr0[3] - destptr4[0];
                        float32x2_t _tmp135b = vsub_f32(_t_33, _t_44);

                        // float t5 = destptr4[1] + destptr4[2];
                        float32x2_t _tmp024c = vadd_f32(_t_55, _t_66);
                        // float t6 = destptr4[1] - destptr4[2];
                        float32x2_t _tmp135c = vsub_f32(_t_55, _t_66);

                        // tmpA[0][m] = destptr0[0] + t1 + t3 + t5 * 32;
                        float32x2_t _output_0 = vadd_f32(_t_00, _tmp024a);
                        _output_0 = vmla_lane_f32(_output_0, _tmp024c, vget_high_f32(_coeff), 1);
                        _output_0 = vadd_f32(_output_0, _tmp024b);

                        // tmpA[2][m] = t1 + t3 * 4 + t5 * 8;
                        float32x2_t _output_2 = vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        _output_2 = vmla_lane_f32(_output_2, _tmp024c, vget_low_f32(_coeff), 1);

                        // tmpA[4][m] = t1 + t3 * 16 + t5 + t5;
                        float32x2_t _output_4 = vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        _output_4 = vadd_f32(_output_4, _tmp024c);
                        _output_4 = vadd_f32(_output_4, _tmp024c);

                        output0[0] = vget_lane_f32(_output_0, 0);
                        output1[0] = vget_lane_f32(_output_0, 1);
                        output0[2] = vget_lane_f32(_output_2, 0);
                        output1[2] = vget_lane_f32(_output_2, 1);
                        output0[4] = vget_lane_f32(_output_4, 0);
                        output1[4] = vget_lane_f32(_output_4, 1);

                        // outptr0[1] = t2 + t4 + t4 + t6 * 16;
                        // t2 + t6 * 16
                        float32x2_t _output_1 = vmla_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        // +t4
                        _output_1 = vadd_f32(_output_1, _tmp135b);
                        // +t4
                        _output_1 = vadd_f32(_output_1, _tmp135b);

                        // outptr0[3] = t2 + t4 * 8 + t6 * 4;
                        // t2 + t4 * 8
                        float32x2_t _output_3 = vmla_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        // + t6*4
                        _output_3 = vmla_lane_f32(_output_3, _tmp135c, vget_low_f32(_coeff), 0);

                        // outptr0[5] = tmp0[7] + t2 + t4 * 32 + t6;
                        // tmp0[7] + t2
                        float32x2_t _output_5 = vadd_f32(_t_77, _tmp135a);
                        // + t4 * 32
                        _output_5 = vmla_lane_f32(_output_5, _tmp135b, vget_high_f32(_coeff), 1);
                        // + t6
                        _output_5 = vadd_f32(_output_5, _tmp135c);

                        output0[1] = vget_lane_f32(_output_1, 0);
                        output1[1] = vget_lane_f32(_output_1, 1);
                        output0[3] = vget_lane_f32(_output_3, 0);
                        output1[3] = vget_lane_f32(_output_3, 1);
                        output0[5] = vget_lane_f32(_output_5, 0);
                        output1[5] = vget_lane_f32(_output_5, 1);

                        t0 += 8 * 2;
                        t1 += 8 * 2;
                        output0 += outW * 2;
                        output1 += outW * 2;
                    }
#else
                    float *t0 = tmpA[0];
                    float *t1 = tmpA[1];

                    int step = dst_tm_w * 2 * tiles * 4 * 4;

                    asm volatile(
                        // loop1
                        "vld1.f32   {d16-d17}, [%2], %21 \n"
                        "vld1.f32   {d18-d19}, [%3], %21 \n"
                        "vld1.f32   {d20-d21}, [%4], %21 \n"
                        "vld1.f32   {d22-d23}, [%5], %21 \n"
                        "vld1.f32   {d24-d25}, [%6], %21 \n"
                        "vld1.f32   {d26-d27}, [%7], %21 \n"
                        "vld1.f32   {d28-d29}, [%8], %21 \n"
                        "vld1.f32   {d30-d31}, [%9], %21 \n"

                        "vtrn.32    q8, q10             \n"
                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        //  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        // q12 = 22  q13 = 66  q14 = 33  q15 = 77
                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"
                        "vswp       d23, d30            \n"

                        // float t1 = destptr0[1] + destptr0[2];
                        "vadd.f32   q2, q10, q12        \n"
                        // float t2 = destptr0[1] - destptr0[2];
                        "vsub.f32   q3, q10, q12        \n"
                        // float t3 = destptr0[3] + destptr4[0];
                        "vadd.f32   q4, q14, q9         \n"
                        // float t4 = destptr0[3] - destptr4[0];
                        "vsub.f32   q5, q14, q9         \n"
                        // float t5 = destptr4[1] + destptr4[2];
                        "vadd.f32   q6, q11, q13        \n"
                        // float t6 = destptr4[1] - destptr4[2];
                        "vsub.f32   q7, q11, q13        \n"

                        // q8 = q8 + q2 = destptr0[0] + t1
                        "vadd.f32   q8, q8, q2          \n"
                        // q8 = q8 + t5 * 32
                        "vmla.f32   q8, q6, %f20[1]     \n"
                        // q8 = q8 + t3
                        "vadd.f32   q8, q8, q4          \n"

                        // q9 = q3 = t2
                        "vmov       q9, q3              \n"
                        "vmla.f32   q9, q7, %f20[0]     \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vadd.f32   q9, q9, q5          \n"

                        "vmov       q10, q2             \n"
                        "vmla.f32   q10, q4, %e20[0]    \n"
                        "vmla.f32   q10, q6, %e20[1]    \n"

                        "vmov       q11, q3             \n"
                        "vmla.f32   q11, q5, %e20[1]    \n"
                        "vmla.f32   q11, q7, %e20[0]    \n"
                        
                        "vmov       q12, q2             \n"
                        "vmla.f32   q12, q4, %f20[0]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vadd.f32   q12, q12, q6        \n"

                        "vadd.f32   q15, q15, q3        \n"
                        "vmla.f32   q15, q5, %f20[1]    \n"
                        "vadd.f32   q15, q15, q7        \n"
                        
                        "vst1.f32   {d16-d17}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d20-d21}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d22-d23}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d24-d25}, [%0]     \n"
                        "sub        %0, %0, #112        \n"

                        "vst1.f32   {d30-d31}, [%1]     \n"
                        "sub        %1, %1, #112        \n"

                        // loop2
                        "vld1.f32   {d16-d17}, [%2], %21 \n"
                        "vld1.f32   {d18-d19}, [%3], %21 \n"
                        "vld1.f32   {d20-d21}, [%4], %21 \n"
                        "vld1.f32   {d22-d23}, [%5], %21 \n"
                        "vld1.f32   {d24-d25}, [%6], %21 \n"
                        "vld1.f32   {d26-d27}, [%7], %21 \n"
                        "vld1.f32   {d28-d29}, [%8], %21 \n"
                        "vld1.f32   {d30-d31}, [%9], %21 \n"

                        "vtrn.32    q8, q10             \n"
                        "vtrn.32    q9, q11             \n"
                        "vtrn.32    q12, q14            \n"
                        "vtrn.32    q13, q15            \n"

                        //  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                        // q12 = 22  q13 = 66  q14 = 33  q15 = 77
                        "vswp       d17, d24            \n"
                        "vswp       d19, d26            \n"
                        "vswp       d21, d28            \n"
                        "vswp       d23, d30            \n"

                        // float t1 = destptr0[1] + destptr0[2];
                        "vadd.f32   q2, q10, q12        \n"
                        // float t2 = destptr0[1] - destptr0[2];
                        "vsub.f32   q3, q10, q12        \n"
                        // float t3 = destptr0[3] + destptr4[0];
                        "vadd.f32   q4, q14, q9         \n"
                        // float t4 = destptr0[3] - destptr4[0];
                        "vsub.f32   q5, q14, q9         \n"
                        // float t5 = destptr4[1] + destptr4[2];
                        "vadd.f32   q6, q11, q13        \n"
                        // float t6 = destptr4[1] - destptr4[2];
                        "vsub.f32   q7, q11, q13        \n"

                        // q8 = q8 + q2 = destptr0[0] + t1
                        "vadd.f32   q8, q8, q2          \n"
                        // q8 = q8 + t5 * 32
                        "vmla.f32   q8, q6, %f20[1]     \n"
                        // q8 = q8 + t3
                        "vadd.f32   q8, q8, q4          \n"

                        // q9 = q3 = t2
                        "vmov       q9, q3              \n"
                        "vmla.f32   q9, q7, %f20[0]     \n"
                        "vadd.f32   q9, q9, q5          \n"
                        "vadd.f32   q9, q9, q5          \n"

                        "vmov       q10, q2             \n"
                        "vmla.f32   q10, q4, %e20[0]    \n"
                        "vmla.f32   q10, q6, %e20[1]    \n"

                        "vmov       q11, q3             \n"
                        "vmla.f32   q11, q5, %e20[1]    \n"
                        "vmla.f32   q11, q7, %e20[0]    \n"
                        
                        "vmov       q12, q2             \n"
                        "vmla.f32   q12, q4, %f20[0]    \n"
                        "vadd.f32   q12, q12, q6        \n"
                        "vadd.f32   q12, q12, q6        \n"

                        "vadd.f32   q15, q15, q3        \n"
                        "vmla.f32   q15, q5, %f20[1]    \n"
                        "vadd.f32   q15, q15, q7        \n"
                        
                        "vst1.f32   {d16-d17}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d18-d19}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d20-d21}, [%0]     \n"
                        "add        %0, %0, #64         \n"

                        "vst1.f32   {d22-d23}, [%1]     \n"
                        "add        %1, %1, #64         \n"

                        "vst1.f32   {d24-d25}, [%0]     \n"
                        "vst1.f32   {d30-d31}, [%1]     \n"

                        : "=r"(t0),            // %0
                        "=r"(t1),            // %1
                        "=r"(destptr00), // %2
                        "=r"(destptr04), // %3
                        "=r"(destptr10), // %4
                        "=r"(destptr14), // %5
                        "=r"(destptr20), // %6
                        "=r"(destptr24), // %7
                        "=r"(destptr30), // %8
                        "=r"(destptr34)  // %9
                        : "0"(t0),
                        "1"(t1),
                        "2"(destptr00),
                        "3"(destptr04),
                        "4"(destptr10),
                        "5"(destptr14),
                        "6"(destptr20),
                        "7"(destptr24),
                        "8"(destptr30),
                        "9"(destptr34),
                        "w"(_coeff), // %20
                        "r"(step)    // %21
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );

                    t0 = tmpA[0];
                    t1 = tmpA[1];

                    float *outptr0 = outptr + (i * 6) * outW + j * 6;
                    float *outptr1 = outptr0 + outW;

                    int stepw = outW * 2 * 4;

                    asm volatile(
                        // loop1
                        
                        // float32x4_t _t0_0123 = vld1q_f32(t0);
                        // float32x4_t _t0_4567 = vld1q_f32(t0 + 4);
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        
                        // float32x4_t _t1_0123 = vld1q_f32(t1);
                        // float32x4_t _t1_4567 = vld1q_f32(t1 + 4);
                        "vld1.f32   {d20-d23}, [%3]     \n"
                        // t0 += 8 * 2 * 4
                        "add        %2, %2, #64         \n"
                        // t1 += 8 * 2 * 4
                        "add        %3, %3, #64         \n"

                        // float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                        // float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);
                        // q8 = 0022, q10 = 1133
                        "vtrn.32    q8, q10             \n"
                        // q9 = 4466, q11 = 5577 
                        "vtrn.32    q9, q11             \n"

                        // float32x2_t _tmp024a = vadd_f32(_t_11, _t_22);
                        "vadd.f32   d4, d20, d17        \n"
                        // float32x2_t _tmp135a = vsub_f32(_t_11, _t_22);
                        "vsub.f32   d5, d20, d17        \n"

                        // float32x2_t _tmp024b = vadd_f32(_t_33, _t_44);
                        "vadd.f32   d6, d21, d18        \n"
                        // float32x2_t _tmp135b = vsub_f32(_t_33, _t_44);
                        "vsub.f32   d7, d21, d18        \n"

                        // float32x2_t _tmp024c = vadd_f32(_t_55, _t_66);
                        "vadd.f32   d8, d22, d19        \n"
                        // float32x2_t _tmp135c = vsub_f32(_t_55, _t_66);
                        "vsub.f32   d9, d22, d19        \n"

                        // tmpA[0][m] = destptr0[0] + t1 + t3 + t5 * 32;
                        // float32x2_t _output_0 = vadd_f32(_t_00, _tmp024a);
                        // _output_0 = vmla_lane_f32(_output_0, _tmp024c, vget_high_f32(_coeff), 1);
                        // _output_0 = vadd_f32(_output_0, _tmp024b);
                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d16, d16, d6        \n"

                        // tmpA[2][m] = t1 + t3 * 4 + t5 * 8;
                        // float32x2_t _output_2 = vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        // _output_2 = vmla_lane_f32(_output_2, _tmp024c, vget_low_f32(_coeff), 1);
                        "vmov       d17, d4             \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"

                        // tmpA[4][m] = t1 + t3 * 16 + t5 + t5;
                        // float32x2_t _output_4 = vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        // _output_4 = vadd_f32(_output_4, _tmp024c);
                        // _output_4 = vadd_f32(_output_4, _tmp024c);
                        "vmov       d18, d4             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d18, d18, d8        \n"

                        // outptr0[1] = t2 + t4 + t4 + t6 * 16;
                        // float32x2_t _output_1 = vmla_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        // +t4
                        // _output_1 = vadd_f32(_output_1, _tmp135b);
                        // +t4
                        // _output_1 = vadd_f32(_output_1, _tmp135b);
                        "vmov       d20, d5             \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vadd.f32   d20, d20, d7        \n"

                        // outptr0[3] = t2 + t4 * 8 + t6 * 4;
                        // t2 + t4 * 8
                        // float32x2_t _output_3 = vmla_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        // + t6*4
                        // _output_3 = vmla_lane_f32(_output_3, _tmp135c, vget_low_f32(_coeff), 0);
                        "vmov       d21, d5             \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        
                        // outptr0[5] = tmp0[7] + t2 + t4 * 32 + t6;
                        // tmp0[7] + t2
                        // float32x2_t _output_5 = vadd_f32(_t_77, _tmp135a);
                        // + t4 * 32
                        // _output_5 = vmla_lane_f32(_output_5, _tmp135b, vget_high_f32(_coeff), 1);
                        // + t6
                        // _output_5 = vadd_f32(_output_5, _tmp135c);
                        "vadd.f32   d22, d23, d5        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"
                        "vadd.f32   d22, d22, d9        \n"

                        // _output_0 -> d16 -> [a1, b1]
                        // _output_1 -> d20 -> [a2, b2]
                        // _output_2 -> d17 -> [c1, d1]
                        // _output_3 -> d21 -> [c2, d2]
                        // _output_4 -> d18 -> [e1, f1]
                        // _output_5 -> d22 -> [e2, f2]
                        // output0[0] = vget_lane_f32(_output_0, 0);
                        // output1[0] = vget_lane_f32(_output_0, 1);
                        // output0[2] = vget_lane_f32(_output_2, 0);
                        // output1[2] = vget_lane_f32(_output_2, 1);
                        // output0[4] = vget_lane_f32(_output_4, 0);
                        // output1[4] = vget_lane_f32(_output_4, 1);
                        // output0[1] = vget_lane_f32(_output_1, 0);
                        // output1[1] = vget_lane_f32(_output_1, 1);
                        // output0[3] = vget_lane_f32(_output_3, 0);
                        // output1[3] = vget_lane_f32(_output_3, 1);
                        // output0[5] = vget_lane_f32(_output_5, 0);
                        // output1[5] = vget_lane_f32(_output_5, 1);

                        // ->
                        // d16 -> [a1, a2]
                        // d17 -> [c1, c2]
                        // d20 -> [b1, b2]
                        // d21 -> [c2, d2]
                        "vtrn.f32   q8, q10             \n"
                        // d18 -> [e1, e2]
                        // d22 -> [f1, f2]
                        "vtrn.f32   d18, d22            \n"

                        // 存储
                        "vst1.f32   {d16-d18}, [%0], %9 \n"
                        "vst1.f32   {d20-d22}, [%1], %9 \n"

                        // loop2

                        // float32x4_t _t0_0123 = vld1q_f32(t0);
                        // float32x4_t _t0_4567 = vld1q_f32(t0 + 4);
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        
                        // float32x4_t _t1_0123 = vld1q_f32(t1);
                        // float32x4_t _t1_4567 = vld1q_f32(t1 + 4);
                        "vld1.f32   {d20-d23}, [%3]     \n"
                        // t0 += 8 * 2 * 4
                        "add        %2, %2, #64         \n"
                        // t1 += 8 * 2 * 4
                        "add        %3, %3, #64         \n"

                        // float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                        // float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);
                        // q8 = 0022, q10 = 1133
                        "vtrn.32    q8, q10             \n"
                        // q9 = 4466, q11 = 5577 
                        "vtrn.32    q9, q11             \n"

                        // float32x2_t _tmp024a = vadd_f32(_t_11, _t_22);
                        "vadd.f32   d4, d20, d17        \n"
                        // float32x2_t _tmp135a = vsub_f32(_t_11, _t_22);
                        "vsub.f32   d5, d20, d17        \n"

                        // float32x2_t _tmp024b = vadd_f32(_t_33, _t_44);
                        "vadd.f32   d6, d21, d18        \n"
                        // float32x2_t _tmp135b = vsub_f32(_t_33, _t_44);
                        "vsub.f32   d7, d21, d18        \n"

                        // float32x2_t _tmp024c = vadd_f32(_t_55, _t_66);
                        "vadd.f32   d8, d22, d19        \n"
                        // float32x2_t _tmp135c = vsub_f32(_t_55, _t_66);
                        "vsub.f32   d9, d22, d19        \n"

                        // tmpA[0][m] = destptr0[0] + t1 + t3 + t5 * 32;
                        // float32x2_t _output_0 = vadd_f32(_t_00, _tmp024a);
                        // _output_0 = vmla_lane_f32(_output_0, _tmp024c, vget_high_f32(_coeff), 1);
                        // _output_0 = vadd_f32(_output_0, _tmp024b);
                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d16, d16, d6        \n"

                        // tmpA[2][m] = t1 + t3 * 4 + t5 * 8;
                        // float32x2_t _output_2 = vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        // _output_2 = vmla_lane_f32(_output_2, _tmp024c, vget_low_f32(_coeff), 1);
                        "vmov       d17, d4             \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"

                        // tmpA[4][m] = t1 + t3 * 16 + t5 + t5;
                        // float32x2_t _output_4 = vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        // _output_4 = vadd_f32(_output_4, _tmp024c);
                        // _output_4 = vadd_f32(_output_4, _tmp024c);
                        "vmov       d18, d4             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d18, d18, d8        \n"

                        // outptr0[1] = t2 + t4 + t4 + t6 * 16;
                        // float32x2_t _output_1 = vmla_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        // +t4
                        // _output_1 = vadd_f32(_output_1, _tmp135b);
                        // +t4
                        // _output_1 = vadd_f32(_output_1, _tmp135b);
                        "vmov       d20, d5             \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vadd.f32   d20, d20, d7        \n"

                        // outptr0[3] = t2 + t4 * 8 + t6 * 4;
                        // t2 + t4 * 8
                        // float32x2_t _output_3 = vmla_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        // + t6*4
                        // _output_3 = vmla_lane_f32(_output_3, _tmp135c, vget_low_f32(_coeff), 0);
                        "vmov       d21, d5             \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        
                        // outptr0[5] = tmp0[7] + t2 + t4 * 32 + t6;
                        // tmp0[7] + t2
                        // float32x2_t _output_5 = vadd_f32(_t_77, _tmp135a);
                        // + t4 * 32
                        // _output_5 = vmla_lane_f32(_output_5, _tmp135b, vget_high_f32(_coeff), 1);
                        // + t6
                        // _output_5 = vadd_f32(_output_5, _tmp135c);
                        "vadd.f32   d22, d23, d5        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"
                        "vadd.f32   d22, d22, d9        \n"

                        // _output_0 -> d16 -> [a1, b1]
                        // _output_1 -> d20 -> [a2, b2]
                        // _output_2 -> d17 -> [c1, d1]
                        // _output_3 -> d21 -> [c2, d2]
                        // _output_4 -> d18 -> [e1, f1]
                        // _output_5 -> d22 -> [e2, f2]
                        // output0[0] = vget_lane_f32(_output_0, 0);
                        // output1[0] = vget_lane_f32(_output_0, 1);
                        // output0[2] = vget_lane_f32(_output_2, 0);
                        // output1[2] = vget_lane_f32(_output_2, 1);
                        // output0[4] = vget_lane_f32(_output_4, 0);
                        // output1[4] = vget_lane_f32(_output_4, 1);
                        // output0[1] = vget_lane_f32(_output_1, 0);
                        // output1[1] = vget_lane_f32(_output_1, 1);
                        // output0[3] = vget_lane_f32(_output_3, 0);
                        // output1[3] = vget_lane_f32(_output_3, 1);
                        // output0[5] = vget_lane_f32(_output_5, 0);
                        // output1[5] = vget_lane_f32(_output_5, 1);

                        // ->
                        // d16 -> [a1, a2]
                        // d17 -> [c1, c2]
                        // d20 -> [b1, b2]
                        // d21 -> [c2, d2]
                        "vtrn.f32   q8, q10             \n"
                        // d18 -> [e1, e2]
                        // d22 -> [f1, f2]
                        "vtrn.f32   d18, d22            \n"

                        // 存储
                        "vst1.f32   {d16-d18}, [%0], %9 \n"
                        "vst1.f32   {d20-d22}, [%1], %9 \n"

                        // loop3

                        // float32x4_t _t0_0123 = vld1q_f32(t0);
                        // float32x4_t _t0_4567 = vld1q_f32(t0 + 4);
                        "vld1.f32   {d16-d19}, [%2]     \n"
                        
                        // float32x4_t _t1_0123 = vld1q_f32(t1);
                        // float32x4_t _t1_4567 = vld1q_f32(t1 + 4);
                        "vld1.f32   {d20-d23}, [%3]     \n"
                        // t0 += 8 * 2 * 4
                        "add        %2, %2, #64         \n"
                        // t1 += 8 * 2 * 4
                        "add        %3, %3, #64         \n"

                        // float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                        // float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);
                        // q8 = 0022, q10 = 1133
                        "vtrn.32    q8, q10             \n"
                        // q9 = 4466, q11 = 5577 
                        "vtrn.32    q9, q11             \n"

                        // float32x2_t _tmp024a = vadd_f32(_t_11, _t_22);
                        "vadd.f32   d4, d20, d17        \n"
                        // float32x2_t _tmp135a = vsub_f32(_t_11, _t_22);
                        "vsub.f32   d5, d20, d17        \n"

                        // float32x2_t _tmp024b = vadd_f32(_t_33, _t_44);
                        "vadd.f32   d6, d21, d18        \n"
                        // float32x2_t _tmp135b = vsub_f32(_t_33, _t_44);
                        "vsub.f32   d7, d21, d18        \n"

                        // float32x2_t _tmp024c = vadd_f32(_t_55, _t_66);
                        "vadd.f32   d8, d22, d19        \n"
                        // float32x2_t _tmp135c = vsub_f32(_t_55, _t_66);
                        "vsub.f32   d9, d22, d19        \n"

                        // tmpA[0][m] = destptr0[0] + t1 + t3 + t5 * 32;
                        // float32x2_t _output_0 = vadd_f32(_t_00, _tmp024a);
                        // _output_0 = vmla_lane_f32(_output_0, _tmp024c, vget_high_f32(_coeff), 1);
                        // _output_0 = vadd_f32(_output_0, _tmp024b);
                        "vadd.f32   d16, d16, d4        \n"
                        "vmla.f32   d16, d8, %f8[1]     \n"
                        "vadd.f32   d16, d16, d6        \n"

                        // tmpA[2][m] = t1 + t3 * 4 + t5 * 8;
                        // float32x2_t _output_2 = vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                        // _output_2 = vmla_lane_f32(_output_2, _tmp024c, vget_low_f32(_coeff), 1);
                        "vmov       d17, d4             \n"
                        "vmla.f32   d17, d6, %e8[0]     \n"
                        "vmla.f32   d17, d8, %e8[1]     \n"

                        // tmpA[4][m] = t1 + t3 * 16 + t5 + t5;
                        // float32x2_t _output_4 = vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                        // _output_4 = vadd_f32(_output_4, _tmp024c);
                        // _output_4 = vadd_f32(_output_4, _tmp024c);
                        "vmov       d18, d4             \n"
                        "vmla.f32   d18, d6, %f8[0]     \n"
                        "vadd.f32   d18, d18, d8        \n"
                        "vadd.f32   d18, d18, d8        \n"

                        // outptr0[1] = t2 + t4 + t4 + t6 * 16;
                        // float32x2_t _output_1 = vmla_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                        // +t4
                        // _output_1 = vadd_f32(_output_1, _tmp135b);
                        // +t4
                        // _output_1 = vadd_f32(_output_1, _tmp135b);
                        "vmov       d20, d5             \n"
                        "vmla.f32   d20, d9, %f8[0]     \n"
                        "vadd.f32   d20, d20, d7        \n"
                        "vadd.f32   d20, d20, d7        \n"

                        // outptr0[3] = t2 + t4 * 8 + t6 * 4;
                        // t2 + t4 * 8
                        // float32x2_t _output_3 = vmla_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                        // + t6*4
                        // _output_3 = vmla_lane_f32(_output_3, _tmp135c, vget_low_f32(_coeff), 0);
                        "vmov       d21, d5             \n"
                        "vmla.f32   d21, d7, %e8[1]     \n"
                        "vmla.f32   d21, d9, %e8[0]     \n"
                        
                        // outptr0[5] = tmp0[7] + t2 + t4 * 32 + t6;
                        // tmp0[7] + t2
                        // float32x2_t _output_5 = vadd_f32(_t_77, _tmp135a);
                        // + t4 * 32
                        // _output_5 = vmla_lane_f32(_output_5, _tmp135b, vget_high_f32(_coeff), 1);
                        // + t6
                        // _output_5 = vadd_f32(_output_5, _tmp135c);
                        "vadd.f32   d22, d23, d5        \n"
                        "vmla.f32   d22, d7, %f8[1]     \n"
                        "vadd.f32   d22, d22, d9        \n"

                        // _output_0 -> d16 -> [a1, b1]
                        // _output_1 -> d20 -> [a2, b2]
                        // _output_2 -> d17 -> [c1, d1]
                        // _output_3 -> d21 -> [c2, d2]
                        // _output_4 -> d18 -> [e1, f1]
                        // _output_5 -> d22 -> [e2, f2]
                        // output0[0] = vget_lane_f32(_output_0, 0);
                        // output1[0] = vget_lane_f32(_output_0, 1);
                        // output0[2] = vget_lane_f32(_output_2, 0);
                        // output1[2] = vget_lane_f32(_output_2, 1);
                        // output0[4] = vget_lane_f32(_output_4, 0);
                        // output1[4] = vget_lane_f32(_output_4, 1);
                        // output0[1] = vget_lane_f32(_output_1, 0);
                        // output1[1] = vget_lane_f32(_output_1, 1);
                        // output0[3] = vget_lane_f32(_output_3, 0);
                        // output1[3] = vget_lane_f32(_output_3, 1);
                        // output0[5] = vget_lane_f32(_output_5, 0);
                        // output1[5] = vget_lane_f32(_output_5, 1);

                        // ->
                        // d16 -> [a1, a2]
                        // d17 -> [c1, c2]
                        // d20 -> [b1, b2]
                        // d21 -> [c2, d2]
                        "vtrn.f32   q8, q10             \n"
                        // d18 -> [e1, e2]
                        // d22 -> [f1, f2]
                        "vtrn.f32   d18, d22            \n"

                        // 存储
                        "vst1.f32   {d16-d18}, [%0], %9 \n"
                        "vst1.f32   {d20-d22}, [%1], %9 \n"


                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(t0),      // %2
                        "=r"(t1)       // %3
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(t0),
                        "3"(t1),
                        "w"(_coeff), // %8
                        "r"(stepw)   // %9
                        : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif

#else
                    float *destptr0 = destptr + (i * w_tm / 8 + j) * dst_tm_w;
                    float *destptr4 = destptr + (i * w_tm / 8 + j + tiles) * dst_tm_w;

                    for(int m = 0; m < 8; m++){

                        float t1 = destptr0[1] + destptr0[2];
                        float t2 = destptr0[1] - destptr0[2];

                        float t3 = destptr0[3] + destptr4[0];
                        float t4 = destptr0[3] - destptr4[0];

                        float t5 = destptr4[1] + destptr4[2];
                        float t6 = destptr4[1] - destptr4[2];

                        tmpA[0][m] = destptr0[0] + t1 + t3 + t5 * 32;
                        tmpA[2][m] = t1 + t3 * 4 + t5 * 8;
                        tmpA[4][m] = t1 + t3 * 16 + t5 + t5;

                        tmpA[1][m] = t2 + t4 + t4 + t6 * 16;
                        tmpA[3][m] = t2 + t4 * 8 + t6 * 4;
                        tmpA[5][m] = destptr4[3] + t2 + t4 * 32 + t6;

                        destptr0 += dst_tm_w * 2 * tiles;
                        destptr4 += dst_tm_w * 2 * tiles;
                    }

                    float *outptr0 = outptr + (i * 6) * outW + j * 6;

                    for(int m = 0; m < 6; m++){

                        const float* tmp0 = tmpA[m];

                        float t1 = tmp0[1] + tmp0[2];
                        float t2 = tmp0[1] - tmp0[2];

                        float t3 = tmp0[3] + tmp0[4];
                        float t4 = tmp0[3] - tmp0[4];

                        float t5 = tmp0[5] + tmp0[6];
                        float t6 = tmp0[5] - tmp0[6];

                        outptr0[0] = tmp0[0] + t1 + t3 + t5 * 32;
                        outptr0[2] = t1 + t3 * 4 + t5 * 8;
                        outptr0[4] = t1 + t3 * 16 + t5 + t5;

                        outptr0[1] = t2 + t4 + t4 + t6 * 16;
                        outptr0[3] = t2 + t4 * 8 + t6 * 4;
                        outptr0[5] = tmp0[7] + t2 + t4 * 32 + t6;

                        outptr0 += outW;
                    }
#endif
                }
            }
        }

        //crop
        for(int cc = 0; cc < outChannel; cc++){
            float *outptr = dest_tm2 + cc * dst_tm_size2;
            float *outptr2 = dest + cc * outHeight * outWidth;
            for(int i = 0; i < outHeight; i++){
                for(int j = 0; j < outWidth; j++){
                    outptr2[0] = outptr[0];
                    outptr2++;
                    outptr++;
                }
                outptr += (outW - outWidth);
            }
        }

        delete [] src_tm;
        delete [] dest_tm;
        delete [] dest_tm2;
        delete [] srcPadding;

    }
}

#endif
