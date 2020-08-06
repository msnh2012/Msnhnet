#include "Msnhnet/layers/cuda/MsnhLocalAvgPoolLayerGPU.h"
namespace Msnhnet
{

__global__ void localAvgPoolNormalKernel(const int n,
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

        float avg           = 0;

        int counter         = 0;

        for (int l = 0; l < kSizeY; ++l)
        {
            for (int m = 0; m < kSizeX; ++m)
            {

                int curHeight   =   heightOffset + i*strideY + l;

                int curWidth    =   widthOffset  + j*strideX + m;

                int index       =   curWidth + width*(curHeight + height*(k + b*channel));

                bool valid      =  (curHeight >=0 && curHeight < height &&
                                    curWidth  >=0 && curWidth  < width);

                if(valid)
                {
                    counter ++;
                    avg         += input[index];
                }
            }
        }
        output[outIndex]        =   avg / counter;
    }
}

void LocalAvgPoolLayerGPU::forwardNormalGPU(const int &width, const int &height, const int &channel, const int &outWidth, const int &outHeight,
                                  const int &outChannel, const int &strideX, const int &strideY, const int &kSizeX, const int kSizeY,
                                  const int &paddingX, const int &paddingY, const int &batch, float *const &input, float *const &output)
{
    size_t n    =   outHeight * outWidth * outChannel * batch;
    localAvgPoolNormalKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n,width,height,
                                                                                           channel,
                                                                                           outHeight,outWidth,
                                                                                           strideX,strideY,
                                                                                           kSizeX,kSizeY,
                                                                                           paddingX,paddingY,
                                                                                           input,output);

    CUDA_CHECK(cudaPeekAtLastError());
}
}
