#define USE_ARM 0

#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolutionSgemm.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    // USE_NEON and __aarch64__ dest[8 * kernelSize, inChannel, outChannel / 8 + (outChannel % 8) / 4 + outChannel % 4]
    // Else dest[4 * kernelSize, inChannel, outChannel/4+outChannel%4]
    void ConvolutionLayerSgemm::convolutionTransformKernel(float *const &kernel, const int &kernelW, const int &kernelH, float* &dest, const int &inChannel,
                            const int &outChannel){
        
        int kernelSize = kernelH * kernelW;
        int ccOutChannel = 0;
        int ccRemainOutChannel = 0;
        int Stride = 0;
#if USE_NEON && __aarch64__
        ccOutChannel = outChannel >> 3;
        ccRemainOutChannel = ccOutChannel << 3;
        Stride = inChannel * (outChannel / 8 + (outChannel % 8) / 4 + outChannel % 4);
        for(int cc = 0; cc < ccOutChannel; cc++){
            int c = cc << 3;
            const float* k0 = kernel + c * inChannel * kernelSize;
            const float* k1 = kernel + (c + 1) * inChannel * kernelSize;
            const float* k2 = kernel + (c + 2) * inChannel * kernelSize;
            const float* k3 = kernel + (c + 3) * inChannel * kernelSize;
            const float* k4 = kernel + (c + 4) * inChannel * kernelSize;
            const float* k5 = kernel + (c + 5) * inChannel * kernelSize;
            const float* k6 = kernel + (c + 6) * inChannel * kernelSize;
            const float* k7 = kernel + (c + 7) * inChannel * kernelSize;

            float* destptr = dest + cc * Stride;
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
        ccRemainOutChannel = ccOutChannel << 2;

        for(int cc = 0;  cc < outChannel; cc ++){
            int c = cc << 2;
            const float* k0 = kernel + c * inChannel * kernelSize;
            const float* k1 = kernel + (c + 1) * inChannel * kernelSize;
            const float* k2 = kernel + (c + 2) * inChannel * kernelSize;
            const float* k3 = kernel + (c + 3) * inChannel * kernelSize;
#if __ARM_NEON && __aarch64__
            Stride = inChannel * (outChannel / 8 + (outChannel % 8) / 4 + outChannel % 4);
            float* destptr = dest + ((c / 8) + (c % 8) / 4) * Stride;
#else
            Stride = inChannel * (outChannel/4+outChannel%4);
            float* destptr = dest + (c / 4) * Stride;
#endif
            for(int i = 0; i < inChannel * kernelSize; i++){
                destptr[0] = k0[0];
                destptr[1] = k1[0];
                destptr[2] = k2[0];
                destptr[3] = k3[0];

                destptr ++ 4;

                k0 += 1;
                k1 += 1;
                k2 += 1;
                k3 += 1;
            }
        }

        for(int cc = ccRemainOutChannel; cc < outChannel; cc++){
            int c = cc;
            const float* k0 = kernel + c * inChannel * kernelSize;
            #if __ARM_NEON && __aarch64__
            Stride = inChannel * (outChannel / 8 + (outChannel % 8) / 4 + outChannel % 4);
            float* destptr = dest + ((c / 8) + (c % 8) / 4 + c % 4) * Stride;
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
    void ConvolutionLayerSgemm::convolutionIm2colSgemm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                            const int &kernelW, const int &kernelH, float* &dest, const int &outWidth, const int &outHeight, const int &outChannel, 
                            const int& StrideH, const int &StrideW){
        // 1. im2col

        float *src_im2col = new float[outWidth * outHeight * kernelH * kernelW * inChannel];
        
        const int Stride = kernelW * kernelH * outHeight * outWidth;
        const int inSize = inHeight * inWidth;
        const int outSize = outHeight * outWidth; 
        const int kernelSize = kernelH * kernelW;

    // inCahnnel x kW x kH
    // outWidth x outHeight

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif

        for(int cc = 0; cc < inChannel; cc++){
            const float *src0 = src + cc * inSize;
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

        // pack 8x8 | idea from ncnn
        // preapare

        const int packWidth = 8 * kernelSize;
        const int packHeight = outSize / 8 + outSize % 8;
        const int packChannel = inChannel;

        float *src_im2col_pack = new float[packHeight * packHeight * packChannel];

        // pack start

        int colCount = outSize >> 3;
        int remainColCount = colCount << 3;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i = 0; i < colCount; i++){
            int newi = i << 3;
            const float *src0 = src_im2col + newi;

            float *packptr = src_im2col_pack + i;

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
            float *packptr = src_im2col_pack + (i / 8 + i % 8);

            for(int j = 0; j < inChannel * kernelSize; j++){
                packptr[0] = src0[0];

                packptr += 1;
                src0 += outSize;

            }
        }

//pack end

// sgemm (int M, int N, int K, float *A, float *B, float *C)
// A (M x N)
// B (N x K)
// C (M x K)

        int M = outChannel;
        int N = outHeight * outWidth;
        int K = kernelSize * inChannel;
        
        int ccOutChannel = outChannel >> 3;
        int ccRemainOutChannel = ccOutChannel << 3;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int cc = 0; cc < ccOutChannel; cc++){
            int c = cc << 3;
            float *destptr0 = dest + c * N;
            float *destptr1 = dest + (c + 1) * N;
            float *destptr2 = dest + (c + 2) * N;
            float *destptr3 = dest + (c + 3) * N;
            float *destptr4 = dest + (c + 4) * N;
            float *destptr5 = dest + (c + 5) * N;
            float *destptr6 = dest + (c + 6) * N;
            float *destptr7 = dest + (c + 7) * N;

            int nn = 0;
            for(; nn + 7 < outHeight * outWidth; nn += 8){
                const float *ptrB = src_im2col_pack + (nn / 8) * ;
                const float *ptrA = kernel_im2col_pack + (c / 8) * kernelSize;
                float sum0[8] = {0};
                float sum1[8] = {0};
                float sum2[8] = {0};
                float sum3[8] = {0};
                float sum4[8] = {0};
                float sum5[8] = {0};
                float sum6[8] = {0};
                float sum7[8] = {0};

                int i = 0;
                for(; i + 7 < K; i += 8){
                    
                }
            }
        }        


        free(src_im2col);
        free(src_im2col_pack);

    }
}

#endif