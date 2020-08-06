#include "Msnhnet/layers/cuda/MsnhYolov3LayerGPU.h"
namespace Msnhnet
{
__global__ void exSigmoidKernel(const int n, float *const input, const int width, const float ratios, const int addGrid)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n)
    {

        if(addGrid == 1)
        {
            input[i] = (1.f/(1.f+expf(-input[i])) + i%width)*ratios;
        }
        else
        {
            input[i] = (1.f/(1.f+expf(-input[i])) + i/width)*ratios;
        }

    }
}

__global__ void sigmoidKernel(const int n, float *const input)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n)
    {
        input[i] = 1.f/(1.f+expf(-input[i]));
    }
}

__global__ void aExpTKernel(const int n, float *const input, const float a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n)
    {
        input[i] = a*expf(input[i]);
    }
}

__global__ void shDataKernel(const int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n)
    {
        printf("%d\n",i);
    }
}

void Yolov3LayerGPU::exSigmoidGpu(const int &n, float *const &input, const int &width, const float &ratios, const int &addGrid)
{
    exSigmoidKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, input, width, ratios, addGrid);
    CUDA_CHECK(cudaPeekAtLastError());
}

void Yolov3LayerGPU::sigmoidGpu(const int &n, float *const &input)
{
    sigmoidKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, input);
    CUDA_CHECK(cudaPeekAtLastError());
}

void Yolov3LayerGPU::aExpTGpu(const int &n, float *const &input, const float &a)
{
    aExpTKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, input, a);
    CUDA_CHECK(cudaPeekAtLastError());
}

}
