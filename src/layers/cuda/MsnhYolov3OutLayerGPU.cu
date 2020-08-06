#include "Msnhnet/layers/cuda/MsnhYolov3OutLayerGPU.h"
namespace Msnhnet
{

__global__ void shuffleDataKernel(const int num, const int wxh, const int chn, float *const allInput, float *const shuffled, const int offset)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < num)
    {
        int m = i % chn;
        i     = i / chn;
        int n = i % wxh;
        i     = i / wxh;
        int k = i;

        shuffled[offset + k*wxh*chn + n*chn + m] = allInput[offset + k*wxh*chn + m*wxh + n];

    }

}

void Yolov3OutLayerGPU::shuffleData(const int &kn, const int &wxh, const int &chn, float *const &allInput, float *const &shuffled, const int &offset)
{
    int num = kn*wxh*chn;
    shuffleDataKernel<<<Cuda::getGrid(num), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(num,wxh,chn,allInput,shuffled,offset);
    CUDA_CHECK(cudaPeekAtLastError());
}

}
