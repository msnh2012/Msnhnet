#define USE_ARM 1
#ifdef USE_ARM
#include "Msnhnet/layers/arm/aarch64/MsnhConvolutionSgemm.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    // pack 8x8
    // shape[c, h, w]: [outChannel / 8 + (outChannel %8)/4 + outChannel%4， 8 * kernelSize， inChannel]
    void ConvolutionLayerArmV8Sgemm::convolutionTransformKernel(float *const &kernel, const int &kernelW, const int &kernelH, float* &dest, const int &inChannel,
                            const int &outChannel){

        int kernelSize = kernelH * kernelW;
        int ccOutChannel = 0;
        int ccRemainOutChannel = 0;
        int Stride = 0;

#if USE_NEON && __aarch64__
        ccOutChannel = outChannel >> 3;
        ccRemainOutChannel = ccOutChannel << 3;

        for(int cc = 0; cc < ccOutChannel; cc++){
            int c = cc << 3;
            const float* k0 = kernel + c * inChannel * kernelSize;
            const float* k1 = kernel + (c + 1) * inChannel * kernelSize;
            const float* k2 = kernel + (c + 2) * inChannel * kernelSize;
            const float* k3 = kernel + (c + 3) * inChannel * kernelSize;
            const float* k4 = kernel + (c + 4) * inChannel * kernelSize;
            const float* k5 = kernel + (c + 5) * inChannel * kernelSize;
            const float* k6 = kernel + (c + 6) * inChannel * kernelSize;
            const float *k7 = kernel + (c + 7) * inChannel * kernelSize;

            Stride = 8 * inChannel * kernelSize;
            float* destptr = dest + (c / 8) * Stride;

            for(int i = 0; i < inChannel * kernelSize; i++){
                destptr[0] = k0[0];
                destptr[1] = k1[0];
                destptr[2] = k2[0];
                destptr[3] = k3[0];
                destptr[4] = k4[0];
                destptr[5] = k5[0];
                destptr[6] = k6[0];
                destptr[7] = k7[0];
                destptr += 8;

                k0 += 1;
                k1 += 1;
                k2 += 1;
                k3 += 1;
                k4 += 1;
                k5 += 1;
                k6 += 1;
                k7 += 1;
            }
        }
#endif            

        ccOutChannel = (outChannel - ccRemainOutChannel) >> 2;

        for(int cc = 0;  cc < ccOutChannel; cc ++){
            int c = cc << 2;
            const float* k0 = kernel + c * inChannel * kernelSize;
            const float* k1 = kernel + (c + 1) * inChannel * kernelSize;
            const float* k2 = kernel + (c + 2) * inChannel * kernelSize;
            const float* k3 = kernel + (c + 3) * inChannel * kernelSize;

            Stride = 4 * inChannel * kernelSize;
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

        ccRemainOutChannel += ccOutChannel << 2;

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
    void ConvolutionLayerArmV8Sgemm::convolutionIm2colSgemm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel, float *const kernel_im2col_pack,
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

        const int kernelPackChannel = outChannel / 8 + (outChannel %8)/4 + outChannel%4;
        const int kernelPackHeight = inChannel;
        const int kernelPackWidth = 8 * kernelSize;

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
                asm volatile(
                    "prfm    pldl1keep, [%0, #256]   \n"
                    "ld1     {v0.4s, v1.4s}, [%0]    \n"
                    "st1     {v0.4s, v1.4s}, [%1]    \n"
                    : "=r"(src0),  // %0
                    "=r"(packptr) // %1
                    : "0"(src0),
                    "1"(packptr)
                    : "cc", "memory", "v0", "v1"
                );
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

        int ccOutChannel = 0;
        int ccRemainOutChannel = 0;

#if USE_NEON && __aarch64__
        ccOutChannel = outChannel >> 3;
        ccRemainOutChannel = ccOutChannel << 3;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = 0; cc < ccOutChannel; cc++){
            int c = cc << 3;
            float *destptr0 = dest + c * outSize;
            float *destptr1 = dest + (c + 1) * outSize;
            float *destptr2 = dest + (c + 2) * outSize;
            float *destptr3 = dest + (c + 3) * outSize;
            float *destptr4 = dest + (c + 4) * outSize;
            float *destptr5 = dest + (c + 5) * outSize;
            float *destptr6 = dest + (c + 6) * outSize;
            float *destptr7 = dest + (c + 7) * outSize;

            int i = 0;
            // N = outHeight*outWidth
            for(; i + 7 < N; i = i + 8){
                const float *ptrB = src_im2col_pack + (i / 8) *  packHeight * packWidth;
                const float *ptrA = kernel_im2col_pack + (c / 8) * kernelPackHeight * kernelPackWidth;
#if __aarch64__
                asm volatile(

                    "eor v16.16b, v16.16b, v16.16b   \n"
                    "eor v17.16b, v17.16b, v17.16b   \n"
                    "eor v18.16b, v18.16b, v18.16b   \n"
                    "eor v19.16b, v19.16b, v19.16b   \n"
                    "eor v20.16b, v20.16b, v20.16b   \n"
                    "eor v21.16b, v22.16b, v23.16b   \n"
                    "eor v24.16b, v24.16b, v24.16b   \n"
                    "eor v25.16b, v25.16b, v25.16b   \n"
                    "eor v26.16b, v26.16b, v26.16b   \n"
                    "eor v27.16b, v27.16b, v27.16b   \n"
                    "eor v28.16b, v28.16b, v28.16b   \n"
                    "eor v29.16b, v29.16b, v29.16b   \n"
                    "eor v30.16b, v30.16b, v30.16b   \n"
                    "eor v31.16b, v31.16b, v31.16b   \n"

                    : "=r"(destptr0), // %0
                    "=r"(destptr1), // %1
                    "=r"(destptr2), // %2
                    "=r"(destptr3), // %3
                    "=r"(destptr4), // %4
                    "=r"(destptr5), // %5
                    "=r"(destptr6), // %6
                    "=r"(destptr7), // %7
                    "=r"(ptrB),      // %8
                    "=r"(ptrA)       // %9
                    : "0"(destptr0),
                    "1"(destptr1),
                    "2"(destptr2),
                    "3"(destptr3),
                    "4"(destptr4),
                    "5"(destptr5),
                    "6"(destptr6),
                    "7"(destptr7),
                    "8"(ptrB),
                    "9"(ptrA),
                    "r"(K)      // %20
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
#else
                float sum0[8] = {0};
                float sum1[8] = {0};
                float sum2[8] = {0};
                float sum3[8] = {0};
                float sum4[8] = {0};
                float sum5[8] = {0};
                float sum6[8] = {0};
                float sum7[8] = {0};
                int j = 0;
                // K = kernelSize * inChannel
                // 同时计算8行，同时在每一列计算8个输出
                for(; j + 7 < K; j = j + 8){
                    for(int n = 0; n < 8; n++){
                        sum0[n] += ptrA[0] * ptrB[n];
                        sum1[n] += ptrA[1] * ptrB[n];
                        sum2[n] += ptrA[2] * ptrB[n];
                        sum3[n] += ptrA[3] * ptrB[n];
                        sum4[n] += ptrA[4] * ptrB[n];
                        sum5[n] += ptrA[5] * ptrB[n];
                        sum6[n] += ptrA[6] * ptrB[n];
                        sum7[n] += ptrA[7] * ptrB[n];
                        ptrA += 8;

                        sum0[n] += ptrA[0] * ptrB[n + 8];
                        sum1[n] += ptrA[1] * ptrB[n + 8];
                        sum2[n] += ptrA[2] * ptrB[n + 8];
                        sum3[n] += ptrA[3] * ptrB[n + 8];
                        sum4[n] += ptrA[4] * ptrB[n + 8];
                        sum5[n] += ptrA[5] * ptrB[n + 8];
                        sum6[n] += ptrA[6] * ptrB[n + 8];
                        sum7[n] += ptrA[7] * ptrB[n + 8];
                        ptrA += 8;

                        sum0[n] += ptrA[0] * ptrB[n + 16];
                        sum1[n] += ptrA[1] * ptrB[n + 16];
                        sum2[n] += ptrA[2] * ptrB[n + 16];
                        sum3[n] += ptrA[3] * ptrB[n + 16];
                        sum4[n] += ptrA[4] * ptrB[n + 16];
                        sum5[n] += ptrA[5] * ptrB[n + 16];
                        sum6[n] += ptrA[6] * ptrB[n + 16];
                        sum7[n] += ptrA[7] * ptrB[n + 16];
                        ptrA += 8;

                        sum0[n] += ptrA[0] * ptrB[n + 24];
                        sum1[n] += ptrA[1] * ptrB[n + 24];
                        sum2[n] += ptrA[2] * ptrB[n + 24];
                        sum3[n] += ptrA[3] * ptrB[n + 24];
                        sum4[n] += ptrA[4] * ptrB[n + 24];
                        sum5[n] += ptrA[5] * ptrB[n + 24];
                        sum6[n] += ptrA[6] * ptrB[n + 24];
                        sum7[n] += ptrA[7] * ptrB[n + 24];
                        ptrA += 8;

                        sum0[n] += ptrA[0] * ptrB[n + 32];
                        sum1[n] += ptrA[1] * ptrB[n + 32];
                        sum2[n] += ptrA[2] * ptrB[n + 32];
                        sum3[n] += ptrA[3] * ptrB[n + 32];
                        sum4[n] += ptrA[4] * ptrB[n + 32];
                        sum5[n] += ptrA[5] * ptrB[n + 32];
                        sum6[n] += ptrA[6] * ptrB[n + 32];
                        sum7[n] += ptrA[7] * ptrB[n + 32];
                        ptrA += 8;

                        sum0[n] += ptrA[0] * ptrB[n + 40];
                        sum1[n] += ptrA[1] * ptrB[n + 40];
                        sum2[n] += ptrA[2] * ptrB[n + 40];
                        sum3[n] += ptrA[3] * ptrB[n + 40];
                        sum4[n] += ptrA[4] * ptrB[n + 40];
                        sum5[n] += ptrA[5] * ptrB[n + 40];
                        sum6[n] += ptrA[6] * ptrB[n + 40];
                        sum7[n] += ptrA[7] * ptrB[n + 40];
                        ptrA += 8;

                        sum0[n] += ptrA[0] * ptrB[n + 48];
                        sum1[n] += ptrA[1] * ptrB[n + 48];
                        sum2[n] += ptrA[2] * ptrB[n + 48];
                        sum3[n] += ptrA[3] * ptrB[n + 48];
                        sum4[n] += ptrA[4] * ptrB[n + 48];
                        sum5[n] += ptrA[5] * ptrB[n + 48];
                        sum6[n] += ptrA[6] * ptrB[n + 48];
                        sum7[n] += ptrA[7] * ptrB[n + 48];
                        ptrA += 8;

                        sum0[n] += ptrA[0] * ptrB[n + 56];
                        sum1[n] += ptrA[1] * ptrB[n + 56];
                        sum2[n] += ptrA[2] * ptrB[n + 56];
                        sum3[n] += ptrA[3] * ptrB[n + 56];
                        sum4[n] += ptrA[4] * ptrB[n + 56];
                        sum5[n] += ptrA[5] * ptrB[n + 56];
                        sum6[n] += ptrA[6] * ptrB[n + 56];
                        sum7[n] += ptrA[7] * ptrB[n + 56];
                        ptrA -= 56;
                    }

                    ptrA += 64;
                    ptrB += 64;
                }

                // K = kernelSize * inChannel
                for(; j < K; j++){
                    for(int n = 0; n < 8; n++){
                        sum0[n] += ptrA[0] * ptrB[n];
                        sum1[n] += ptrA[1] * ptrB[n];
                        sum2[n] += ptrA[2] * ptrB[n];
                        sum3[n] += ptrA[3] * ptrB[n];
                        sum4[n] += ptrA[4] * ptrB[n];
                        sum5[n] += ptrA[5] * ptrB[n];
                        sum6[n] += ptrA[6] * ptrB[n];
                        sum7[n] += ptrA[7] * ptrB[n];
                    }
                    ptrA += 8;
                    ptrB += 8;
                }

                for(int n = 0; n < 8; n++){
                    destptr0[n] = sum0[n];
                    destptr1[n] = sum1[n];
                    destptr2[n] = sum2[n];
                    destptr3[n] = sum3[n];
                    destptr4[n] = sum0[n];
                    destptr5[n] = sum1[n];
                    destptr6[n] = sum2[n];
                    destptr7[n] = sum3[n];
                }

#endif
                destptr0 += 8;
                destptr1 += 8;
                destptr2 += 8;
                destptr3 += 8;
                destptr4 += 8;
                destptr5 += 8;
                destptr6 += 8;
                destptr7 += 8;
            }

            for(; i < N; i++){
                const float *ptrB = src_im2col_pack + (i / 8 + i % 8) *  packHeight * packWidth;
                const float *ptrA = kernel_im2col_pack + (c / 8) * kernelPackHeight * kernelPackWidth;
#if USE_NEON
#else
                float sum0 = 0.f;
                float sum1 = 0.f;
                float sum2 = 0.f;
                float sum3 = 0.f;
                float sum4 = 0.f;
                float sum5 = 0.f;
                float sum6 = 0.f;
                float sum7 = 0.f;

                for(int j = 0; j < K; j++){
                    sum0 += ptrA[0] * ptrB[0];
                    sum1 += ptrA[1] * ptrB[0];
                    sum2 += ptrA[2] * ptrB[0];
                    sum3 += ptrA[3] * ptrB[0];
                    sum4 += ptrA[4] * ptrB[0];
                    sum5 += ptrA[5] * ptrB[0];
                    sum6 += ptrA[6] * ptrB[0];
                    sum7 += ptrA[7] * ptrB[0];

                    ptrA += 8;
                    ptrB += 1;
                }

                destptr0[0] = sum0;
                destptr1[0] = sum1;
                destptr2[0] = sum2;
                destptr3[0] = sum3;
                destptr4[0] = sum4;
                destptr5[0] = sum5;
                destptr6[0] = sum6;
                destptr7[0] = sum7;
#endif
                destptr0++;
                destptr1++;
                destptr2++;
                destptr3++;
                destptr4++;
                destptr5++;
                destptr6++;
                destptr7++;                
            }
        }

#endif
        ccOutChannel = (outChannel - ccRemainOutChannel) >> 2;

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

        ccRemainOutChannel += ccOutChannel << 2;
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
