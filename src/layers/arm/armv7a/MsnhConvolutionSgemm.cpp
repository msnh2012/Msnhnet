#ifdef USE_ARM
#include "Msnhnet/layers/arm/armv7a/MsnhConvolutionSgemm.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    // pack 4x4
    // shape[c, h, w]: [outChannel / 4 + outChannel %4， 4 * kernelSize， inChannel]
    void ConvolutionLayerSgemm::convolutionTransformKernel(float *const &kernel, const int &kernelW, const int &kernelH, float* &dest, const int &inChannel,
                            const int &outChannel){
        
        int kernelSize = kernelH * kernelW;
        int ccOutChannel = 0;
        int ccRemainOutChannel = 0;
        int Stride = 0;
        
        ccOutChannel = outChannel >> 2;
        ccRemainOutChannel = ccOutChannel << 2;

        for(int cc = 0;  cc < ccOutChannel; cc ++){
            int c = cc << 2;
            const float* k0 = kernel + c * inChannel * kernelSize;
            const float* k1 = kernel + (c + 1) * inChannel * kernelSize;
            const float* k2 = kernel + (c + 2) * inChannel * kernelSize;
            const float* k3 = kernel + (c + 3) * inChannel * kernelSize;

            Stride = 4 * kernelSize * inChannel;
            float* destptr = dest + (c / 4) * Stride;

            for(int i = 0; i < inChannel * kernelSize; i++){
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

        for(int cc = ccRemainOutChannel; cc < outChannel; cc++){
            int c = cc;
            const float* k0 = kernel + c * inChannel * kernelSize;
            Stride = 4 * kernelSize * inChannel;
            float* destptr = dest + (c / 4 + c % 4) * Stride;
            for(int i = 0; i < inChannel * kernelSize; i++){
                destptr[0] = k0[0];
                destptr += 1;
                k0 += 1;
            }
        }
    }
    void ConvolutionLayerSgemm::convolutionIm2colSgemm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel, float *const kernel_im2col_pack,
                            const int &kernelW, const int &kernelH, float* &dest, const int &outWidth, const int &outHeight, const int &outChannel, 
                            const int& StrideH, const int &StrideW){
        // 1. im2col
        // src_im2col : width=outWidth * outHeight, height=kernelH * kernelW * inChannel
        float *src_im2col = new float[outWidth * outHeight * kernelH * kernelW * inChannel];
        
        const int Stride = kernelW * kernelH * outHeight * outWidth;
        //const int inSize = inHeight * inWidth;
        const int outSize = outHeight * outWidth; 
        const int kernelSize = kernelH * kernelW;

    // inCahnnel x kW x kH
    // outWidth x outHeight

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = 0; cc < inChannel; cc++){
            const float *src0 = src + cc * inHeight * inWidth;
            int dst_idx = Stride * cc;
            for(int i = 0; i < kernelH; i++){
                for(int j = 0; j < kernelW; j++){
                    for(int x = 0; x < outHeight; x++){
                        for(int y = 0; y < outWidth; y++){
                            int row = x * StrideH + i;
                            int col = y * StrideW + j;
                            int ori_idx = row * inWidth + col;
                            src_im2col[dst_idx] = src0[ori_idx];
                            dst_idx++;      
                        }
                    }
                }
            }
        }

        // printf("Im2Col: \n");
        // for(int i=0; i<outWidth * outHeight * kernelH * kernelW * inChannel; i++){
        //     printf("%f ", src_im2col[i]);
        //     if(i>0&&i%(kernelH * kernelW * inChannel)==0){
        //         printf("\n");
        //     }
        // }

        // pack 8x8
        // preapare


        const int packChannel = outSize / 8 + outSize % 8;
        const int packHeight = inChannel;    
        const int packWidth = 8 * kernelSize;

        int kernelPackChannel = outChannel / 4 + outChannel % 4;
        const int kernelPackHeight = inChannel;
        const int kernelPackWidth = 4 * kernelSize;

        float *src_im2col_pack = new float[packHeight * packWidth * packChannel];

        // pack start

        int colCount = outSize >> 3;
        int remainColCount = colCount << 3;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i = 0; i < colCount; i++){
            int newi = i << 3;
            const float *src0 = src_im2col;

            src0 += newi;

            float *packptr = src_im2col_pack + i * packHeight * packWidth;

            for(int j = 0; j < inChannel * kernelSize; j ++){
#if USE_NEON
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    "pld        [%0, #256]          \n"
                    "vld1.f32   {d0-d3}, [%0]       \n"
                    "vst1.f32   {d0-d3}, [%1]       \n"
                    : "=r"(src0),  // %0
                    "=r"(packptr) // %1
                    : "0"(src0),
                    "1"(packptr)
                    : "memory", "q0", "q1"
                );
#endif

#else
                packptr[0] = src0[0];
                packptr[1] = src0[1];
                packptr[2] = src0[2];
                packptr[3] = src0[3];
                packptr[4] = src0[4];
                packptr[5] = src0[5];
                packptr[6] = src0[6];
                packptr[7] = src0[7];
#endif
                packptr += 8;
                src0 += outSize;

            }

        }

// pack tail

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i = remainColCount; i < outSize; i++){
            const float *src0 = src_im2col;
            src0 += i;
            float *packptr = src_im2col_pack + (i / 8 + i % 8) * packHeight * packWidth;

            for(int j = 0; j < inChannel * kernelSize; j++){
                packptr[0] = src0[0];

                packptr += 1;
                src0 += outSize;

            }
        }
        // printf("Pack: \n");
        // for(int i = 0; i < packHeight * packWidth * packChannel; i++){
        //     printf("%f ", src_im2col_pack[i]);
        //     if(i>0&&i%(packHeight * packWidth)==0){
        //         printf("\n");
        //     }
        // }

//pack end

// sgemm (int M, int N, int K, float *A, float *B, float *C)
// A (M x K)
// B (K x N)
// C (M x N)

        //int M = outChannel;
        int N = outHeight * outWidth;
        int K = kernelSize * inChannel;
        
        int ccOutChannel = outChannel >> 2;
        int ccRemainOutChannel = ccOutChannel << 2;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif

        for(int cc = 0; cc < ccOutChannel; cc++){
            int c = cc << 2;
            float *destptr0 = dest + c * outSize;
            float *destptr1 = dest + (c + 1) * outSize;
            float *destptr2 = dest + (c + 2) * outSize;
            float *destptr3 = dest + (c + 3) * outSize;

            int i = 0;
            // N = outHeight*outWidth
            for(; i + 7 < N; i = i+8){
                const float *ptrB = src_im2col_pack + (i / 8) *  packHeight * packWidth;
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                const float *ptrA = kernel_im2col_pack + (c / 4) * kernelPackHeight * kernelPackWidth;
#endif

#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else

                asm volatile(
                    "veor       q1, q0, q0           \n"
                    "vdup.f32   q8,    d2[0]         \n"
                    "vdup.f32   q9,    d2[0]         \n"
                    "vdup.f32   q10,   d2[0]         \n"
                    "vdup.f32   q11,   d2[0]         \n"
                    "vdup.f32   q12,   d2[0]         \n"
                    "vdup.f32   q13,   d2[0]         \n"
                    "vdup.f32   q14,   d2[0]         \n"
                    "vdup.f32   q15,   d2[0]         \n"
                    
                    // r4 = K >> 2
                    "lsr         r4, %12, #2        \n"
                    // 如果nn等于0，使用beq进行循环跳转，即跳转到循环1 
                    "cmp         r4, #0             \n"
                    "beq         loop1              \n"
                    // for(; nn != 0; nn--) && nn = K >> 2
                    "loop0:                         \n" 
                    // kernel q0-q3
                    "pld        [%5, #512]          \n"
                    "vldm       %5!, {d0-d7}        \n" 
                    // input  q4-q7
                    "pld        [%4, #512]          \n"
                    "vldm       %4!, {d8-d15}       \n"

                    //calc
                    // sum0[n] += ptrA[0] * ptrB[n];
                    "vmla.f32   q8, q4, d0[0]       \n"
                    // sum1[n] += ptrA[1] * ptrB[n];
                    "vmla.f32   q9, q5, d0[0]       \n"
                    // sum0[n] += ptrA[0] * ptrB[n + 8];
                    "vmla.f32   q10, q4, d0[1]      \n"
                    // sum1[n] += ptrA[1] * ptrB[n + 8];
                    "vmla.f32   q11, q5, d0[1]      \n"
                    // sum0[n] += ptrA[0] * ptrB[n + 16];
                    "vmla.f32   q12, q4, d1[0]      \n" 
                    // sum1[n] += ptrA[1] * ptrB[n + 16];
                    "vmla.f32   q13, q5, d1[0]      \n"
                    // sum0[n] += ptrA[0] * ptrB[n + 24];
                    "vmla.f32   q14, q4, d1[1]      \n" 
                    // sum1[n] += ptrA[1] * ptrB[n + 24];
                    "vmla.f32   q15, q5, d1[1]      \n"

                    // sum2[n] += ptrA[2] * ptrB[n];
                    "vmla.f32   q8, q6, d2[0]       \n" 
                    // sum3[n] += ptrA[3] * ptrB[n];
                    "vmla.f32   q9, q7, d2[0]       \n"
                    // sum2[n] += ptrA[2] * ptrB[n + 8];
                    "vmla.f32   q10, q6, d2[1]      \n" 
                    // sum3[n] += ptrA[3] * ptrB[n + 8];
                    "vmla.f32   q11, q7, d2[1]      \n"
                    // sum2[n] += ptrA[2] * ptrB[n + 16];
                    "vmla.f32   q12, q6, d3[0]      \n" 
                    // sum3[n] += ptrA[3] * ptrB[n + 16];
                    "vmla.f32   q13, q7, d3[0]      \n"
                    // sum2[n] += ptrA[2] * ptrB[n + 24];
                    "vmla.f32   q14, q6, d3[1]      \n" 
                    // sum3[n] += ptrA[3] * ptrB[n + 24];
                    "vmla.f32   q15, q7, d3[1]      \n"

                    // ptrA += 4x4
                    "pld        [%4, #512]          \n"
                    "vldm       %4!, {d8-d15}       \n"

                    // sum0[n] += ptrA[0] * ptrB[n + 32];
                    "vmla.f32   q8, q4, d4[0]       \n" 
                    // sum1[n] += ptrA[1] * ptrB[n + 32];
                    "vmla.f32   q9, q5, d4[0]       \n"
                    // sum0[n] += ptrA[0] * ptrB[n + 40];
                    "vmla.f32   q10, q4, d4[1]      \n" 
                    // sum1[n] += ptrA[1] * ptrB[n + 40];
                    "vmla.f32   q11, q5, d4[1]      \n"
                    // sum0[n] += ptrA[0] * ptrB[n + 48];
                    "vmla.f32   q12, q4, d5[0]      \n" 
                    // sum1[n] += ptrA[1] * ptrB[n + 48];
                    "vmla.f32   q13, q5, d5[0]      \n"
                    // sum0[n] += ptrA[0] * ptrB[n + 56];
                    "vmla.f32   q14, q4, d5[1]      \n" 
                    // sum1[n] += ptrA[1] * ptrB[n + 56];
                    "vmla.f32   q15, q5, d5[1]      \n"

                    // sum2[n] += ptrA[2] * ptrB[n + 32];
                    "vmla.f32   q8, q6, d6[0]       \n" 
                    // sum3[n] += ptrA[3] * ptrB[n + 32];
                    "vmla.f32   q9, q7, d6[0]       \n"
                    // sum2[n] += ptrA[2] * ptrB[n + 40];
                    "vmla.f32   q10, q6, d6[1]      \n" 
                    // sum3[n] += ptrA[3] * ptrB[n + 40];
                    "vmla.f32   q11, q7, d6[1]      \n"
                    // sum2[n] += ptrA[2] * ptrB[n + 48];
                    "vmla.f32   q12, q6, d7[0]      \n"
                    // sum3[n] += ptrA[3] * ptrB[n + 48];
                    "vmla.f32   q13, q7, d7[0]      \n"
                    // sum2[n] += ptrA[2] * ptrB[n + 56];
                    "vmla.f32   q14, q6, d7[1]      \n" 
                    // sum3[n] += ptrA[3] * ptrB[n + 56];
                    "vmla.f32   q15, q7, d7[1]      \n"

                    "subs        r4, r4, #1         \n"
                    // 第一个for循环的结束，nn>0
                    "bne         loop0             \n" 

                    // 开始写第二个for循环
                    "loop1:                         \n"
                    // K = kernelSize * inChannel * 4
                    // K >> 2 == inChannel>>2 = inChannel & 3
                    // 计算完之后进行第三个for循环进行最后的赋值
                    "and         r4, %12, #3        \n"
                    "cmp         r4, #0             \n"
                    "beq         loop3              \n"

                    "loop2:                         \n" 
                    // kernel q0 && ptrA += 4
                    // q0 = [d0, d1] = [ptrA[0], ptrA[1], ptrA[2], ptrA[3]]
                    "pld        [%5, #128]          \n"
                    "vld1.f32   {d0-d1}, [%5]!      \n"
                    // input q4, q5 && ptrB += 8
                    // q4, q5 = [d8, d9, d10, d11] = [ptrB[0], ..., ptrB[7]]
                    "pld        [%4, #256]          \n"
                    "vld1.f32   {d8-d11}, [%4]!     \n"

                    // for(int n = 0; n < 8; n++){
                    //    sum0[n] += ptrA[0] * ptrB[n];
                    //    sum1[n] += ptrA[1] * ptrB[n];
                    //    sum2[n] += ptrA[2] * ptrB[n];
                    //    sum3[n] += ptrA[3] * ptrB[n];
                    // }
                    "vmla.f32   q8, q4, d0[0]       \n" 
                    "vmla.f32   q9, q5, d0[0]       \n"
                    "vmla.f32   q10, q4, d0[1]      \n"
                    "vmla.f32   q11, q5, d0[1]      \n"
                    "vmla.f32   q12, q4, d1[0]      \n" 
                    "vmla.f32   q13, q5, d1[0]      \n"
                    "vmla.f32   q14, q4, d1[1]      \n" 
                    "vmla.f32   q15, q5, d1[1]      \n"

                    "subs        r4, r4, #1         \n"
                    "bne         loop2             \n"

                    // 完成赋值
                    "loop3:                         \n" 
                    "vst1.f32    {d16-d19}, [%0]    \n"
                    "vst1.f32    {d20-d23}, [%1]    \n"
                    "vst1.f32    {d24-d27}, [%2]    \n"
                    "vst1.f32    {d28-d31}, [%3]    \n"


                    : "=r"(destptr0), // %0
                    "=r"(destptr1), // %1
                    "=r"(destptr2), // %2
                    "=r"(destptr3), // %3
                    "=r"(ptrB),      // %4
                    "=r"(ptrA)       // %5
                    : "0"(destptr0),
                    "1"(destptr1),
                    "2"(destptr2),
                    "3"(destptr3),
                    "4"(ptrB),
                    "5"(ptrA),
                    "r"(K)      // %12
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif

#else
                float sum0[8] = {0};
                float sum1[8] = {0};
                float sum2[8] = {0};
                float sum3[8] = {0};
                int j = 0;
                // K = kernelSize * inChannel
                // 同时计算4行，同时在每一列计算8个输出
                for(; j + 7 < K; j = j + 8){
                    for(int n = 0; n < 8; n++){
                        sum0[n] += ptrA[0] * ptrB[n];
                        sum1[n] += ptrA[1] * ptrB[n];
                        sum2[n] += ptrA[2] * ptrB[n];
                        sum3[n] += ptrA[3] * ptrB[n];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 8];
                        sum1[n] += ptrA[1] * ptrB[n + 8];
                        sum2[n] += ptrA[2] * ptrB[n + 8];
                        sum3[n] += ptrA[3] * ptrB[n + 8];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 16];
                        sum1[n] += ptrA[1] * ptrB[n + 16];
                        sum2[n] += ptrA[2] * ptrB[n + 16];
                        sum3[n] += ptrA[3] * ptrB[n + 16];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 24];
                        sum1[n] += ptrA[1] * ptrB[n + 24];
                        sum2[n] += ptrA[2] * ptrB[n + 24];
                        sum3[n] += ptrA[3] * ptrB[n + 24];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 32];
                        sum1[n] += ptrA[1] * ptrB[n + 32];
                        sum2[n] += ptrA[2] * ptrB[n + 32];
                        sum3[n] += ptrA[3] * ptrB[n + 32];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 40];
                        sum1[n] += ptrA[1] * ptrB[n + 40];
                        sum2[n] += ptrA[2] * ptrB[n + 40];
                        sum3[n] += ptrA[3] * ptrB[n + 40];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 48];
                        sum1[n] += ptrA[1] * ptrB[n + 48];
                        sum2[n] += ptrA[2] * ptrB[n + 48];
                        sum3[n] += ptrA[3] * ptrB[n + 48];
                        ptrA += 4;

                        sum0[n] += ptrA[0] * ptrB[n + 56];
                        sum1[n] += ptrA[1] * ptrB[n + 56];
                        sum2[n] += ptrA[2] * ptrB[n + 56];
                        sum3[n] += ptrA[3] * ptrB[n + 56];
                        ptrA -= 28;

                    }

                    ptrA += 32;
                    ptrB += 64;

                }
                // K = kernelSize * inChannel * 4
                // 如果是pack4x4那么末尾一定是4的倍数
                for(; j < K; j++){
                    for(int n = 0; n < 8; n++){
                        sum0[n] += ptrA[0] * ptrB[n];
                        sum1[n] += ptrA[1] * ptrB[n];
                        sum2[n] += ptrA[2] * ptrB[n];
                        sum3[n] += ptrA[3] * ptrB[n];
                    }
                    ptrA += 4;
                    ptrB += 8;
                }

                for(int n = 0; n < 8; n++){
                    destptr0[n] = sum0[n];
                    destptr1[n] = sum1[n];
                    destptr2[n] = sum2[n];
                    destptr3[n] = sum3[n];
                }

#endif
                destptr0 += 8;
                destptr1 += 8;
                destptr2 += 8;
                destptr3 += 8;

            }

            // N = outHeight*outWidth
            // 拖尾部分，在列方向上只能逐个计算
            for(; i < N; i++){
                const float *ptrB = src_im2col_pack + (i / 8 + i % 8) *  packHeight * packWidth;
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                const float *ptrA = kernel_im2col_pack + (c / 4) * kernelPackHeight * kernelPackWidth;
#endif
                
#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm volatile(
                    
                    "veor       q12, q12, q12       \n"

                    // r4 = K >> 2
                    "lsr         r4, %12, #2        \n" 
                    "cmp         r4, #0             \n"
                    "beq         loop6              \n"

                    // veor 异或，寄存器值初始化为0
                    // q8, q9, q10, q11 = sum0, sum1, sum2, sum3
                    
                    "veor       q8, q8, q8          \n"
                    "veor       q9, q9, q9          \n"
                    "veor       q10, q10, q10       \n"
                    "veor       q11, q11, q11       \n"

                    "loop5:                          \n"
                    // kernel: [q0,q1,q2,q3]
                    "pld        [%5, #512]          \n"
                    "vldm       %5!, {d0-d7}        \n" 

                    // input: [q4, q5]
                    "pld        [%4, #128]          \n"
                    "vld1.f32   {d8-d9}, [%4]!      \n" 

                    // 
                    "vmla.f32   q8, q0, d8[0]       \n" 
                    "vmla.f32   q9, q1, d8[1]       \n"
                    "vmla.f32   q10, q2, d9[0]      \n"
                    "vmla.f32   q11, q3, d9[1]      \n" 

                    "subs        r4, r4, #1         \n"
                    "bne         loop5              \n"

                    "vadd.f32   q8, q8, q9          \n"
                    "vadd.f32   q10, q10, q11       \n"
                    "vadd.f32   q8, q8, q10         \n"
                    "vadd.f32   q12, q12, q8        \n"

                    "loop6:                         \n"
                    // r4 = remain = inChannel&3
                    "and         r4, %12, #3        \n" 
                    "cmp         r4, #0             \n"
                    "beq         loop8             \n"

                    "loop7:                         \n" 
                    "pld        [%5, #128]          \n"
                    "vld1.f32   {d0-d1}, [%5]!      \n"
                    "pld        [%4, #32]           \n"
                    "vld1.f32   {d8[],d9[]}, [%4]!  \n"

                    "subs       r4, r4, #1          \n"

                    "vmla.f32   q12, q0, q4         \n"
                    "bne         loop7             \n"

                    "loop8:                         \n" 
                    "vst1.f32    {d24[0]}, [%0]     \n"
                    "vst1.f32    {d24[1]}, [%1]     \n"
                    "vst1.f32    {d25[0]}, [%2]     \n"
                    "vst1.f32    {d25[1]}, [%3]     \n"

                    : "=r"(destptr0), // %0
                    "=r"(destptr1), // %1
                    "=r"(destptr2), // %2
                    "=r"(destptr3), // %3
                    "=r"(ptrB),      // %4
                    "=r"(ptrA)       // %5
                    : "0"(destptr0),
                    "1"(destptr1),
                    "2"(destptr2),
                    "3"(destptr3),
                    "4"(ptrB),
                    "5"(ptrA),
                    "r"(K)      // %12
                    : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12"
                );
#endif

#else 
                float sum0 = 0;
                float sum1 = 0;
                float sum2 = 0;
                float sum3 = 0;
                // K = kernelSize * inChannel * 4
                for(int j = 0; j < K; j++){
                    sum0 += ptrA[0] * ptrB[0];
                    sum1 += ptrA[1] * ptrB[0];
                    sum2 += ptrA[2] * ptrB[0];
                    sum3 += ptrA[3] * ptrB[0];

                    ptrA += 4;
                    ptrB += 1;
                }

                destptr0[0] = sum0;
                destptr1[0] = sum1;
                destptr2[0] = sum2;
                destptr3[0] = sum3;

#endif
                destptr0++;
                destptr1++;
                destptr2++;
                destptr3++;
            }

        }


        //tail
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = ccRemainOutChannel; cc < outChannel; cc++){
            int c = cc;
            float *destptr0 = dest + c * outSize;
            int i = 0;
            for(; i + 7 < N; i = i + 8){
                const float *ptrB = src_im2col_pack + (i / 8) *  packHeight * packWidth;
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                const float *ptrA = kernel_im2col_pack + (c / 4 + c % 4) * kernelPackHeight * kernelPackWidth;
#endif

#if 0

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm vilatile();
#endif

#else
                float sum[8]= {0};
                int j = 0;
                // K = kernelSize * inChannel * 4
                // 只计算一行，在列方向同时计算8个输出
                for(; j + 7 < K; j = j + 8){
                    for(int n = 0; n < 8; n++){
                        sum[n] += ptrA[0] * ptrB[n];
                        sum[n] += ptrA[1] * ptrB[n + 8];
                        sum[n] += ptrA[2] * ptrB[n + 16];
                        sum[n] += ptrA[3] * ptrB[n + 24];
                        sum[n] += ptrA[4] * ptrB[n + 32];
                        sum[n] += ptrA[5] * ptrB[n + 40];
                        sum[n] += ptrA[6] * ptrB[n + 48];
                        sum[n] += ptrA[7] * ptrB[n + 56];
                    }

                    ptrA += 8;
                    ptrB += 64;
                }
                // 拖尾部分
                for(; j < K; j++){
                    for(int n = 0; n < 8; n++){
                        sum[n] += ptrA[0] * ptrB[n];
                    }

                    ptrA += 1;
                    ptrB += 8;
                }

                for(int n = 0; n < 8; n++){
                    destptr0[n] = sum[n];
                }

#endif
                destptr0 += 8;

            }

            for(; i < N; i++){
                const float *ptrB = src_im2col_pack + (i / 8 + i % 8) *  packHeight * packWidth;
#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                const float *ptrA = kernel_im2col_pack + (c / 4 + c % 4) * kernelPackHeight * kernelPackWidth;
#endif
                int j = 0;        
                float sum = 0;

                for(; j < K; j++){
                    sum += ptrA[0] * ptrB[0];

                    ptrA += 1;
                    ptrB += 1;
                }

                destptr0[0] = sum;

                destptr0++;

            }
        }

        delete [] src_im2col;
        delete [] src_im2col_pack;
        src_im2col = nullptr;
        src_im2col_pack = nullptr;
    }
}

#endif
