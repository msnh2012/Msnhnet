#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{

__global__ void fp32ToFp16Kernel(float *const fp32, const size_t size, __half *const fp16)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < size)
    {
        fp16[i] = __float2half(fp32[i]);
    }
}

void  Cuda::fp32ToFp16(float *const &fp32, const size_t &size, float *const &fp16)
{
    fp32ToFp16Kernel<<<Cuda::getGrid(size), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(fp32, size, (half*)fp16);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void fp16ToFp32Kernel(__half *const fp16, size_t size, float *const fp32)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < size)
    {
        fp32[i] = __half2float(fp16[i]);
    }
}

void  Cuda::fp16ToFp32(float * const &fp16 , const size_t &size, float *const &fp32)
{
    fp16ToFp32Kernel<<<Cuda::getGrid(size), Cuda::blockThread, 0, Cuda::getCudaStream()>>>((half*)fp16, size, fp32);
    CUDA_CHECK(cudaPeekAtLastError());
}

}

