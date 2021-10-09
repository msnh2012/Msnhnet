#include "Msnhnet/layers/cuda/MsnhClipLayerGPU.h"
namespace Msnhnet
{

__global__ void clipKernel(const int num, const int outHeight, const int outWidth, const int outChannel,
                                const float min, const float max,
                                float *const input, float *const output)
{

    int idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(idx < num)
    {
        if(input[idx]>max)
            output[idx] = max;
        else if(input[idx]<min)
            output[idx] = min;
        else
            output[idx] = input[idx];

    }
}
void ClipLayerGPU::forwardNormalGPU(const int &batch,  const int &outChannel, const int &outHeight, const int &outWidth,
                               const float &min,  const float &max,
                               float *const &input, float * const &output)
{
    size_t n = outHeight * outWidth * outChannel * batch;
    clipKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, outHeight, outWidth, outChannel, min, max, input, output);
    CUDA_CHECK(cudaPeekAtLastError());
}

}
