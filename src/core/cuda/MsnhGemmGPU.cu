
#include "Msnhnet/core/cuda/MsnhGemmGPU.h"
namespace Msnhnet
{
































































__global__ void im2ColExKernel(const int n, float *input,
                                  const int height, const int width,
                                  const int kernelH, const int kernelW,
                                  const int padH, const int padW,
                                  const int strideH,const int strideW,
                                  const int dilationH, const int dilationW,
                                  const int outColH, const int outColW,
                                  float *output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < n; index += blockDim.x * gridDim.x)
    {
        const int hIndex    =   index / outColW;
        const int hCol      =   hIndex % outColH;
        const int wCol      =   index % outColW;
        const int cIm       =   hIndex / outColH;


        const int cCol      =   cIm * kernelH *kernelW;
        const int hOff      =   hCol * strideH - padH;
        const int wOff      =   wCol * strideW - padW;

        float* dataColPtr   =   output;
        dataColPtr         +=   (cCol * outColH + hCol)*outColW + wCol;

        float* dataInPtr    =   input;
        dataInPtr          +=   (cIm * height + hOff) * width + wOff;

        for (int i = 0; i < kernelH; ++i)
        {
            for (int j = 0; j < kernelW; ++j)
            {
                int hIm     =   hOff + i * dilationH;
                int wIm     =   wOff + j * dilationW;

                if(hIm >= 0 && wIm >= 0 && hIm < height && wIm < width)
                {
                    *dataColPtr =  dataInPtr[i * dilationH * width + j * dilationW ];
                }
                else
                {
                    *dataColPtr = 0;
                }

                dataColPtr  += outColH*outColW;
            }
        }

    }
}

void GemmGPU::gpuIm2ColEx(float *input, const int &channelNum,
                  const int &height, const int &width,
                  const int &kernelH, const int &kernelW,
                  const int &padH, const int &padW,
                  const int &strideH,  const int &strideW,
                  const int &dilationH, const int &dilationW,
                  float *output)
{










































    int outColH     =  (height + 2*padH - (dilationH * (kernelH - 1) + 1)) / strideH + 1;
    int outColW     =  (width  + 2*padW - (dilationW * (kernelW - 1) + 1)) / strideW + 1;
    int numKernel   =  channelNum * outColH * outColW;
    int blocks      =  (numKernel + Cuda::blockThread - 1)/Cuda::blockThread; 


    im2ColExKernel<<<blocks,Cuda::blockThread,0,Cuda::getCudaStream()>>>(numKernel,input,height,width,kernelH,kernelW,padH,padW,strideH,strideW,dilationH,dilationW,outColH,outColW,output);
    CUDA_CHECK(cudaPeekAtLastError());
}






































































__global__ void im2colKernel(const int n, float *const input,
                              const int height, const int width,
                              const int kSize,
                              const int stride,
                              const int padding,
                              const int outColH, const int outColW,
                              float *const output)
{
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for(; index < n; index += blockDim.x*gridDim.x)
    {
        int wOut    = index % outColW;
        int hIndex  = index / outColW;
        int hOut    = hIndex % outColH;
        int chIn    = hIndex / outColH;
        int chOut   = chIn * kSize * kSize;
        int hIn     = hOut * stride - padding;
        int wIn     = wOut * stride - padding;
        float* dataColPtr       =   output;
        dataColPtr += (chOut * outColH + hOut) * outColW + wOut;
        const float* dataInPtr  =   input;
        dataInPtr  += (chIn * height + hIn) * width + wIn;
        for (int i = 0; i < kSize; ++i)
        {
            for (int j = 0; j < kSize; ++j)
            {
                int h = hIn + i;
                int w = hIn + j;

                if(h >= 0 && w >= 0 && h < height && w < width)
                {
                    *dataColPtr =  dataInPtr[i * width + j];
                }
                else
                {
                    *dataColPtr = 0;
                }

                dataColPtr  += outColH*outColW;
            }
        }
    }


}































void GemmGPU::gpuIm2col(float *const &input, const int &channelNum,
                     const int &height, const int &width,
                     const int &kSize,
                     const int &stride,
                     const int &padding,
                     float *const &output)
{
    int outColH     =   (height + 2 * padding - kSize) / stride + 1;
    int outColW     =   (width  + 2 * padding - kSize) / stride + 1;
    int numKernel   =   channelNum * outColH * outColW;
    int blocks      =   (numKernel + Cuda::blockThread - 1)/Cuda::blockThread; 


    im2colKernel<<<blocks,Cuda::blockThread,0,Cuda::getCudaStream()>>>(numKernel,input,height,width,kSize,stride,padding,outColH,outColW,output);
    CUDA_CHECK(cudaPeekAtLastError());
}


void GemmGPU::gpuGemm(const int &TA, const int &TB, const int &M, const int &N, const int &K, const float &ALPHA,
                   float * const &A, const int &lda,
                   float * const &B, const int &ldb,
                   const float &BETA,
                   float * const &C, const int &ldc)
{
    cublasHandle_t handle = Cuda::getBlasHandle();
    CUBLAS_CHECK(cublasSetStream(handle, Cuda::getCudaStream()));
    CUBLAS_CHECK(cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),(TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B, ldb, A, lda, &BETA, C, ldc));
}

}









