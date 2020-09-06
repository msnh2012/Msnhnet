#include "Msnhnet/layers/cuda/MsnhGlobalAvgPoolLayerGPU.h"
namespace Msnhnet
{
__global__ void globalAvgPoolNormalKernel(const int n, const int width, const int height, const int channel, float *const input, float *const output)
{

    int index   = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(index < n)
    {
        int k   = index % channel;
        index   = index / channel;

        int b   = index;

        int outIndex        = k + b * channel;
        output[outIndex]    = 0;

        for (int i = 0; i < width * height; ++i)
        {
            int inIndex     = i + height*width*(k + b*channel);
            output[outIndex]+= input[inIndex];
        }
        output[outIndex]    /= height*width;
    }
}

void GlobalAvgPoolLayerGPU::forwardNormalGPU(const int &width, const int &height, const int &channel, const int &batch, float *const &input, float *const &output)
{
    size_t n    =   channel * batch;
    globalAvgPoolNormalKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n,width,height,channel,input,output);

    CUDA_CHECK(cudaPeekAtLastError());
}

}
