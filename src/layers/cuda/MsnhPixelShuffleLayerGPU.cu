#include "Msnhnet/layers/cuda/MsnhPixelShuffleLayerGPU.h"

namespace Msnhnet
{

__global__ void pixShuffleKernel(const int num, const int outHeight, const int outWidth, const int outChannel,
                                const int height, const int width, const int channel,
                                const int factor, float *const input, float *const output)
{

    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(index < num)
    {
        int w = index % outWidth;
        index = index / outWidth;
        int h = index % outHeight;
        index = index / outHeight;
        int c = index % outChannel;
        index = index / outChannel;
        int b = index;

        output[b*outChannel*outWidth*outHeight + c*outWidth*outHeight + h*outWidth + w]
        =
        input[b*channel*width*height + c*factor*width*height + h*width + w%factor*width*height + w/factor];

    }
}

void PixelShuffleLayerGPU::forwardNormalGPU(const int &batch, const int &outChannel, const int &outHeight, const int &outWidth,
                                        const int &height, const int &width, const int &channel,
                                        const int &factor,float *const &input, float *const &output)
{
    size_t n = outHeight * outWidth * outChannel * batch;
    pixShuffleKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, outHeight, outWidth, outChannel, height, width, channel, factor, input, output);
    CUDA_CHECK(cudaPeekAtLastError());
}

}
