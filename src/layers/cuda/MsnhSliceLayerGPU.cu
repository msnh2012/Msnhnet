#include "Msnhnet/layers/cuda/MsnhSliceLayerGPU.h"
namespace Msnhnet
{

__global__ void sliceKernel(const int num, const int outChannel, const int outHeight, const int outWidth,
                             const int height, const int width, const int channel,
                             const int start0, const int step0,
                             const int start1, const int step1,
                             const int start2, const int step2,
                             float *const input, float * const output)
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

        output[b*outChannel*outHeight*outWidth + c*outHeight*outWidth + h*outWidth + w]
                =
         input[b*channel*height*width + (c*step0 + start0)*height*width +
                                        (h*step1 + start1)*width +
                                        (w*step2 + start2)];
    }
}

void SliceLayerGPU::forwardNormalGPU(const int &batch, const int &outChannel, const int &outHeight, const int &outWidth,
                                     const int &height, const int &width, const int &channel,
                                     const int &start0, const int &step0,
                                     const int &start1, const int &step1,
                                     const int &start2, const int &step2,
                                     float *const &input, float * const &output)
{
    size_t n = outHeight * outWidth * outChannel * batch;
     sliceKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, outChannel, outHeight, outWidth, height, width, channel, start0, step0, start1, step1, start2, step2, input, output);
    CUDA_CHECK(cudaPeekAtLastError());
}

}
