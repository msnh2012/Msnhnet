#include "Msnhnet/layers/cuda/MsnhYoloLayerGPU.h"
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

__global__ void exSigmoidV5Kernel(const int n, float *const input, const int width, const float ratios, const int addGrid)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n)
    {
        if(addGrid == 1)
        {
            input[i] = (2.f/(1.f+expf(-input[i])) - 0.5f + i%width)*ratios;
        }
        else
        {
            input[i] = (2.f/(1.f+expf(-input[i])) - 0.5f + i/width)*ratios;
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

__global__ void aPowSigmoidKernel(const int n, float *const input, const float a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n)
    {
        input[i] = a*powf(2.f/(1.f+expf(-input[i])),2.f);
    }
}

void YoloLayerGPU::exSigmoidGpu(const int &n, float *const &input, const int &width, const float &ratios, const int &addGrid)
{
    exSigmoidKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, input, width, ratios, addGrid);
    CUDA_CHECK(cudaPeekAtLastError());
}

void YoloLayerGPU::exSigmoidV5Gpu(const int &n, float *const &input, const int &width, const float &ratios, const int &addGrid)
{
    exSigmoidV5Kernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, input, width, ratios, addGrid);
    CUDA_CHECK(cudaPeekAtLastError());
}

void YoloLayerGPU::sigmoidGpu(const int &n, float *const &input)
{
    sigmoidKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, input);
    CUDA_CHECK(cudaPeekAtLastError());
}

void YoloLayerGPU::aPowSigmoid(const int &n, float *const &input, const float &a)
{
    aPowSigmoidKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, input, a);
    CUDA_CHECK(cudaPeekAtLastError());
}

void YoloLayerGPU::aExpTGpu(const int &n, float *const &input, const float &a)
{
    aExpTKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, input, a);
    CUDA_CHECK(cudaPeekAtLastError());
}

}
