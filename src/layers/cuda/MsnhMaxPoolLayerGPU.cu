#include "Msnhnet/layers/cuda/MsnhMaxPoolLayerGPU.h"
namespace Msnhnet
{

__global__ void maxpoolDepthKernel(const int n, const int width, const int height, const int channel,
                                     const int outChannel, const int batch, float *const input, float *const output)
{

    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(index < n)
    {
        int j   = index % width;
        index   = index / width;
        int i   = index % height;
        index   = index / height;

        int b   = index % batch;

        for (int g = 0; g < outChannel; ++g)
        {
            int outIndex    = j + width*(i + height*(g + outChannel*b));
            float max       = -FLT_MAX;

            for (int k = g; k < channel; k+=outChannel)
            {
                int inIndex = j + width*(i + height*(k + channel*b));
                float val   = input[inIndex];
                max         = (val > max)?val:max;
            }
            output[outIndex] = max;
        }
    }
}

__global__ void maxpoolNormalKernel(const int n,
                                     const int width, const int height,
                                     const int channel,
                                     const int outWidth, const int outHeight,
                                     const int strideX, const int strideY,
                                     const int kSizeX, const int kSizeY,
                                     const int paddingX, const int paddingY,
                                     float *const input, float *const output)
{

    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(index < n)
    {
        int j = index % outWidth; 

        index = index / outWidth;
        int i = index % outHeight;

        index = index / outHeight;

        int k = index % channel;  

        index = index / channel;  

        int b = index;

        int widthOffset     =   -(paddingX + 1)/2;
        int heightOffset    =   -(paddingY + 1)/2;

        int outIndex        =   j + outWidth*(i + outHeight*(k + channel*b));
        float max           =   -INFINITY;

        for (int l = 0; l < kSizeY; ++l)
        {
            for (int m = 0; m < kSizeX; ++m)
            {

                int curHeight   =   heightOffset + i*strideY + l;

                int curWidth    =   widthOffset  + j*strideX + m;

                int idx         =   curWidth + width*(curHeight + height*(k + b*channel));

                bool valid      =  (curHeight >=0 && curHeight < height &&
                                    curWidth  >=0 && curWidth  < width);

                float value     =  (valid != 0)? input[idx] : -INFINITY;

                max             =  (value > max) ? value : max;
            }
        }

        output[outIndex]        =   max;

    }

}

void MaxPoolLayerGPU::forwardDepthGPU(const int &width, const int &height, const int &channel, const int &outWidth, const int &outHeight,
                                    const int &outChannel, const int &batch, float *const &input, float *const &output)
{
    size_t n = outHeight * outWidth * 1 * batch;
    maxpoolDepthKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, width, height, channel, outChannel, batch, input, output);
    CUDA_CHECK(cudaPeekAtLastError());
}

void MaxPoolLayerGPU::forwardNormalGPU(const int &width, const int &height, const int &channel, const int &outWidth, const int &outHeight,
                                  const int &outChannel, const int &strideX, const int &strideY, const int &kSizeX, const int kSizeY,
                                  const int &paddingX, const int &paddingY, const int &batch, float *const &input, float *const &output)
{

    size_t n    =   outHeight * outWidth * outChannel * batch;
    maxpoolNormalKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n,width,height,
                                                                                           channel,
                                                                                           outHeight,outWidth,
                                                                                           strideX,strideY,
                                                                                           kSizeX,kSizeY,
                                                                                           paddingX,paddingY,
                                                                                           input,output);
    CUDA_CHECK(cudaPeekAtLastError());
}
}
