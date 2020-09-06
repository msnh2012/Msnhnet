#include "Msnhnet/layers/cuda/MsnhPermuteLayerGPU.h"
namespace Msnhnet
{

__global__ void permuteKernel(const int num, const int outHeight, const int outWidth, const int outChannel,
                                const int height, const int width, const int channel,
                                const int dim0, const int dim1, const int dim2,
                                float *const input, float *const output)
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

        int cc = 0;
        int hh = 0;
        int ww = 0;

        if(dim0 == 0 && dim1 == 1 &&dim2 == 2)
        {
            cc = c;
            hh = h;
            ww = w;
        }
        else if(dim0 == 0 && dim1 == 2 &&dim2 == 1)
        {
            cc = c;
            hh = w;
            ww = h;
        }
        else if(dim0 == 1 && dim1 == 0 &&dim2 == 2)
        {
            cc = h;
            hh = c;
            ww = w;
        }
        else if(dim0 == 1 && dim1 == 2 &&dim2 == 0)
        {
            cc = w;
            hh = c;
            ww = h;
        }
        else if(dim0 == 2 && dim1 == 0 &&dim2 == 1)
        {
            cc = h;
            hh = w;
            ww = c;
        }
        else if(dim0 == 2 && dim1 == 1 &&dim2 == 0)
        {
            cc = w;
            hh = h;
            ww = c;
        }

        output[b*outChannel*outWidth*outHeight + c*outWidth*outHeight + h*outWidth + w]
        =
        input[b*channel*width*height + cc*width*height + hh*width + ww];

    }
}

void PermuteLayerGPU::forwardNormalGPU(const int &batch, const int &outChannel, const int &outHeight, const int &outWidth,
                                        const int &height, const int &width, const int &channel,
                                        const int &dim0, const int &dim1, const int &dim2,
                                        float *const &input, float *const &output)
{
    size_t n = outHeight * outWidth * outChannel * batch;
    permuteKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, outHeight, outWidth, outChannel, height, width, channel, dim0, dim1, dim2, input, output);
    CUDA_CHECK(cudaPeekAtLastError());
}

}
