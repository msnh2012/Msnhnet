#include "Msnhnet/layers/cuda/MsnhPaddingLayerGPU.h"
namespace Msnhnet
{

__global__ void paddingKernel(const int num, const int outHeight, const int outWidth, const int outChannel,
                              const int height, const int width, const int channel,
                              const int top, const int left,
                              const float paddingVal,
                              float *const input, float *const output)
{

    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(index < num)
    {
        int n = index % outWidth;
        index = index / outWidth;
        int m = index % outHeight;
        index = index / outHeight;
        int j = index % outChannel;
        index = index / outChannel;
        int i = index;

        float val = 0;

        if(m < top || m>=(height+top))
        {
            val = paddingVal;
        }
        else
        {
            if(n < left || n >=(width+left))
            {
                val = paddingVal;
            }
            else
            {
                val = input[i*channel*height*width + j*height*width + (m - top)*width + (n-left)];
            }
        }
        output[i*outChannel*outHeight*outWidth + j*outHeight*outWidth + m*outWidth + n] = val;
    }
}

void PaddingLayerGPU::forwardNormalGPU(const int &batch, const int &outChannel, const int &outHeight, const int &outWidth,
                                    const int &height, const int &width, const int &channel,
                                    const int &top, const int &left,
                                    const float &paddingVal,
                                    float * const &input, float * const &output)
{
    size_t n = outHeight * outWidth * outChannel * batch;
    paddingKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, outHeight, outWidth, outChannel, height, width, channel, top, left, paddingVal, input, output);
    CUDA_CHECK(cudaPeekAtLastError());
}

}
