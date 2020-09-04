#define USE_ARM 0

#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolutionSgemm.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    // pack 4x4 dest[4 * kernelSize, inChannel, outChannel / 4 + outChanne l %4]
    void ConvolutionLayerSgemm::convolutionTransformKernel(float *const &kernel, const int &kernelW, const int &kernelH, float* &dest, const int &inChannel,
                            const int &outChannel){
        
        int kernelSize = kernelH * kernelW;
        int ccOutChannel = 0;
        int ccRemainOutChannel = 0;
        int Stride = 0;
        
        ccOutChannel = (outChannel - ccRemainOutChannel) >> 2;
        ccRemainOutChannel = ccOutChannel << 2;

        for(int cc = 0;  cc < ccOutChannel; cc ++){
            int c = cc << 2;
            const float* k0 = kernel + c * inChannel * kernelSize;
            const float* k1 = kernel + (c + 1) * inChannel * kernelSize;
            const float* k2 = kernel + (c + 2) * inChannel * kernelSize;
            const float* k3 = kernel + (c + 3) * inChannel * kernelSize;
#if __aarch64__
           throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
            Stride = inChannel * (outChannel/4+outChannel%4);
            float* destptr = dest + (c / 4) * Stride;
#endif
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
#if __aarch64__
            throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
            Stride = inChannel * (outChannel/4+outChannel%4);
            float* destptr = dest + (c / 4 + c % 4) * Stride;
#endif
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
            const float *src0 = src + cc * kernelH * kernelW * inChannel;
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

        // pack 8x8
        // preapare


        const int packChannel = 8 * kernelSize;
        const int packHeight = inChannel;    
        const int packWidth = outSize / 8 + outSize % 8;

        int kernelPackChannel = 4 * kernelSize;
        const int kernelPackHeight = inChannel;
        const int kernelPackWidth = outChannel / 4 + outChannel % 4;

        float *src_im2col_pack = new float[packHeight * packWidth * packChannel];

        // pack start

        int colCount = outSize >> 3;
        int remainColCount = colCount << 3;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i = 0; i < colCount; i++){
            int newi = i << 3;
            const float *src0 = src_im2col + newi;

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
                    : "=r"(img0),  // %0
                    "=r"(tmpptr) // %1
                    : "0"(img0),
                    "1"(tmpptr)
                    : "memory", "q0", "q1");
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
            const float *src0 = src_im2col + i;
            float *packptr = src_im2col_pack + (i / 8 + i % 8) * packHeight * packWidth;

            for(int j = 0; j < inChannel * kernelSize; j++){
                packptr[0] = src0[0];

                packptr += 1;
                src0 += outSize;

            }
        }

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

                );
#endif

#else
                float sum0[8] = {0};
                float sum1[8] = {0};
                float sum2[8] = {0};
                float sum3[8] = {0};
                int j = 0;
                // K = kernelSize * inChannel
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
                // K = kernelSize * inChannel
                // 如果是pack4那么末尾一定是4的倍数
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

            // N = outChannel
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
                asm volatile();
#endif

#else 
                float sum0 = 0;
                float sum1 = 0;
                float sum2 = 0;
                float sum3 = 0;
                for(int j = 0; j < K; j++){
                    sum0 += ptrA[0] * ptrB[0];
                    sum1 += ptrA[1] * ptrB[1];
                    sum2 += ptrA[2] * ptrB[2];
                    sum3 += ptrA[3] * ptrB[3];

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
        for(int cc = remainColCount; cc < outChannel; cc++){
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

#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else
                asm vilatile();
#endif

#else
                float sum[8]= {0};
                int j = 0;
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

    }
}

#endif