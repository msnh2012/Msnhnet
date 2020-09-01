#define USE_ARM 0

#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolutionSgemm.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
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
        int remainColCount = outSize - (colCount << 3);

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




        free(src_im2col);
        free(src_im2col_pack);

    }
}

#endif