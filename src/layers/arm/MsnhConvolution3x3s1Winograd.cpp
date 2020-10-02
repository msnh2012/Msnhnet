#define USE_ARM 1
#define __aarch64__ 1
#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolution3x3s1Winograd.h"
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

                        :
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
#endif
                    k00 += 4;
                    k01 += 4;
                    k10 += 4;
                    k11 += 4;
                    k20 += 4;
                    k21 += 4;
                    k30 += 4;
                    k31 += 4;
                    ktm2 += 32;

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
                        "=r"(k01),  // %2
                        "=r"(k10),  // %3
                        "=r"(k11)  // %4

                        :
                        : "cc", "memory", "q0", "q1");
#endif

#else
                    for(int j = 0; j < 4; j++){
                        ktm2[0 + j] = k00[j];
                        ktm2[4 + j] = k10[j];
                        ktm2[8 + j] = k20[j];
                        ktm2[12 + j] = k30[j];
                    }
#endif
                    k00 += 4;
                    k10 += 4;
                    k20 += 4;
                    k30 += 4;
                    ktm2 += 16;
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

                        :
                        : "cc", "memory", "q0");
#endif

#else
                    for(int j = 0; j < 4; j++){
                        ktm2[j] = k00[j];
                    }
#endif
                    k00 += 4;
                    ktm2 += 4;
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
                    float *r1 = r0 + PadWidth * 2;
                    float *r2 = r0 + PadWidth * 3;
                    //no swap intrinsic, so based on vtrnq_f32 and combile intrinsic
#if __aarch64__
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
                        vst1q_f32(&tmpV[7][m], t6);

                        

                    }
#else
                   
#endif

                   
#else

#endif

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

                }
            }
        }

        delete [] srcPadding;

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
            }

            for(; q < inChannel; q++){
                const float *r0 = src_tm + q * src_tm_size;
                float* destptr0 = dest0;
                float *destptr1 = dest1;
                float *destptr2 = dest2;
                float *destptr3 = dest3;

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
                    for(int i = 0; i < tiles; i++){
                        for(int m = 0; m < 4; m++){
                            destptr0[m] += r0[m] * ktm[m];
                        }

                        r0 += 4;
                        destptr0 += 4;
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

    }
}

#endif
