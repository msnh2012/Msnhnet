#include "Msnhnet/layers/cuda/MsnhConvolutionalLayerGPU.h"

namespace Msnhnet
{

__global__ void convBnKernel(const int n, const int outChannel, const int outWxH, float *const gpuScales,
                             float *const gpuRollMean, float *const gpuRollVariance, float *const gpuBiases, float *const gpuOutput)
{
    int index   = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(index < n)
    {
        int i = index % outWxH;
        index = index / outWxH;
        int c = index % outChannel;
        index = index / outChannel;
        int b = index;

        int id = b*outChannel*outWxH + c*outWxH + i;

        gpuOutput[id]  = gpuScales[c]*(gpuOutput[id] - gpuRollMean[c])/sqrtf(gpuRollVariance[c] + 0.00001f) + gpuBiases[c];
    }
}

void ConvolutionalLayerGPU::convBn(const int &batch, const int &outChannel, const int &outHeight, const int &outWidth, float* const &gpuScales,
                                     float *const &gpuRollMean, float *const &gpuRollVariance, float *const &gpuBiases, float *const &gpuOutput)
{
    int num     = batch*outChannel*outWidth*outHeight;
    int outWxH  = outHeight*outWidth;

    convBnKernel<<<Cuda::getGrid(num), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(num, outChannel, outWxH, gpuScales,
                                                                                      gpuRollMean, gpuRollVariance, gpuBiases, gpuOutput);

    CUDA_CHECK(cudaPeekAtLastError());
}
}
