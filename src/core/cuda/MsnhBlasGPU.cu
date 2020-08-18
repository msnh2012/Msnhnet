#include "Msnhnet/core/cuda/MsnhBlasGPU.h"
namespace Msnhnet
{

__global__ void copySimpleKernel(const int size,  float * const src, float * const dst)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size)
        dst[index] = src[index];
}

void BlasGPU::gpuSimpleCopy(const int &size,  float * const &src, float * const &dst)
{
    const int num_blocks = size / Cuda::blockThread + 1;
    copySimpleKernel<<<num_blocks, Cuda::blockThread, 0, Cuda::getCudaStream()>>>(size, src, dst);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void  copyKernel(const int n, float *const x, const int incX, float *const y, const int incY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i<n)
    {
        y[i*incY] = x[i*incX] ;
    }
}

void BlasGPU::gpuCopy(const int &n,  float *const x, const int &incX, float *const y, const int &incY)
{
    copyKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n,x,incX,y,incY);
    CUDA_CHECK(cudaPeekAtLastError());
}

void BlasGPU::gpuMemcpy(void *const &dst, void *const &src, const int &size)
{
    CUDA_CHECK(cudaMemcpyAsync(dst,src,size,cudaMemcpyDefault,Cuda::getCudaStream()));
    CUDA_CHECK(cudaPeekAtLastError());
}
__global__ void fillKernel(const int n, const int alpha, float *const x, const int step)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i<n)
    {
        x[i*step] = alpha;
    }
}
void BlasGPU::gpuFill(const int &n, const float &alpha, float *const &x, const int &step)
{
    fillKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n,alpha,x,step);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void  axpyKernel(const int n, const float alpha, float *const x, const int incX, float *const y, const int incY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i<n)
    {
        y[i*incY] += alpha*x[i*incX] ;
    }
}

void BlasGPU::gpuAxpy(const int &n, const float &alpha, float * const x, const int &incX, float * const y, const int &incY)
{
    axpyKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n,alpha,x,incX,y,incY);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void  arithmeticKernel(const Arithmetic type, const int n, float * const x, const int stepX, float * const y, const int stepY, float *out, const int stepOut)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i<n)
    {
        switch (type)
        {
        case ARITH_ADD:
            out[i*stepOut] = x[i*stepX] + y[i*stepY];
            break;
        case ARITH_SUB:
            out[i*stepOut] = x[i*stepX] - y[i*stepY];
            break;
        case ARITH_SUB_INV:
            out[i*stepOut] = y[i*stepY] - x[i*stepX];
            break;
        case ARITH_MUL:
            out[i*stepOut] = x[i*stepX] * y[i*stepY];
            break;
        case ARITH_DIV:
            float tmp1 = (y[i*stepY]==0?0.000001:y[i*stepY]);
            out[i*stepOut] = x[i*stepX] / tmp1 ;
            break;
        case ARITH_DIV_INV:
            float tmp2 = (x[i*stepX]==0?0.000001:x[i*stepX]);
            out[i*stepOut] = y[i*stepX] / tmp2;
            break;
        }
    }
}

void BlasGPU::gpuArithmetic(const Arithmetic &type, const int &n, float * const &x, const int &stepX, float * const &y, const int &stepY, float *out, const int &stepOut)
{
    arithmeticKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(type, n, x, stepX, y, stepY, out, stepOut);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void  arithmeticConstKernel(const Arithmetic type, const int n, float * const x, const int stepX, const float alpha, float *out, const int stepOut)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i<n)
    {
        switch (type)
        {
        case ARITH_ADD:
            out[i*stepOut] = x[i*stepX] + alpha;
            break;
        case ARITH_SUB:
            out[i*stepOut] = x[i*stepX] - alpha;
            break;
        case ARITH_SUB_INV:
            out[i*stepOut] = alpha - x[i*stepX];
            break;
        case ARITH_MUL:
            out[i*stepOut] = x[i*stepX] * alpha;
            break;
        case ARITH_DIV:
            float tmp1 = (alpha==0?0.000001:alpha);
            out[i*stepOut] = x[i*stepX] / tmp1 ;
            break;
        case ARITH_DIV_INV:
            float tmp2 = (x[i*stepX]==0?0.000001:x[i*stepX]);
            out[i*stepOut] = alpha / tmp2;
            break;
        }
    }
}

void BlasGPU::gpuArithmetic(const Arithmetic &type, const int &n, float * const &x, const int &stepX, const float &alpha, float *out, const int &stepOut)
{
    arithmeticConstKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(type, n, x, stepX, alpha, out, stepOut);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void  scaleKernel(const int n, const float alpha, float * const x, const int step)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i<n)
    {
        x[i*step] *= alpha;
    }
}

void BlasGPU::gpuScale(const int &n, const float &alpha, float * const &x, const int &step)
{
    scaleKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n,alpha,x,step);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void meanKernel(float *const x, const int batch, const int filters, const int outSize, float *const mean)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    float scale = 1.f/(batch*outSize);

    if(i >= filters) return;

    mean[i] = 0;

    for (int j = 0; j < batch; ++j)
    {
        for (int k = 0; k < outSize; ++k)
        {
            int index = j*filters*outSize + i*outSize + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}
void BlasGPU::gpuMean(float *const &x, const int &batch, const int &filters, const int &outSize, float *const &mean)
{
    meanKernel<<<Cuda::getGrid(filters), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(x,batch,filters,outSize,mean);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void varianceKernel(float *const x, float *const mean, const int batch, const int filters,
                                const int outSize, float *const variance)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    float scale = 1.f/(batch*outSize - 1);

    if(i >= filters) return;

    variance[i] = 0;

    for (int j = 0; j < batch; ++j)
    {
        for (int k = 0; k < outSize; ++k)
        {
            int index = j*filters*outSize + i*outSize + k;
            variance[i] += powf((x[index] - mean[i]),2);
        }
    }

    variance[i]*=scale;
}

void BlasGPU::gpuVariance(float *const &x, float *const &mean, const int &batch, const int &filters,
                       const int &outSize, float *const &variance)
{
    varianceKernel<<<Cuda::getGrid(filters), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(x,mean,batch,filters,outSize,variance);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void normKernel(const int n, float *const x, float *const mean, float *const variance,
                            const int batch, const int filters, const int outSize)
{

    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(index < n)
    {
        int f = (index / outSize ) % filters;
        x[index] = (x[index] - mean[f])/(sqrtf(variance[f] + 0.00001f));
    }
}

void BlasGPU::gpuNorm(float *const &x, float *const &mean, float *const &variance,
                    const int &batch, const int &filters, const int &outSize)
{
    const int size  = batch*filters*outSize;
    normKernel<<<Cuda::getGrid(size), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(size, x, mean, variance, batch, filters, outSize);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void smoothL1Kernel(const int n, float *const pred, float *const truth, float *const delta, float *const error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n)
    {
        float diff          = truth[i] - pred[i];
        float absDiff       = abs(diff);

        if(absDiff < 1)
        {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else
        {
            error[i] = 2*absDiff - 1;
            delta[i] = (diff < 0)?-1:1;
        }
    }
}

void BlasGPU::gpuSmoothL1(const int &n, float *const &pred, float *const &truth, float *const &delta, float *const &error)
{
    smoothL1Kernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, pred, truth, delta, error);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void l1Kernel(const int n, float *const pred, float *const truth, float *const delta, float *const error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n)
    {
        float diff = truth[i] - pred[i];
        error[i] = abs(diff);
        delta[i] = (diff > 0) ? 1 : -1;
    }
}

void BlasGPU::gpuL1(const int &n, float *const &pred, float *const &truth, float *const &delta, float *const &error)
{
    l1Kernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, pred, truth, delta, error);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void l2Kernel(const int n, float *const pred, float *const truth, float *const delta, float *const error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n)
    {
        float diff = truth[i] - pred[i];
        error[i] = diff*diff;
        delta[i] = diff;
    }
}

void BlasGPU::gpuL2(const int &n, float *const &pred, float *const &truth, float *const &delta, float *const &error)
{
    l2Kernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, pred, truth, delta, error);
    CUDA_CHECK(cudaPeekAtLastError());
}

__device__ void _softmaxKernel(float *input, int num, float temperature, int stride, float *output)
{
    float sum = 0;
    float largest = -INFINITY;
    for (int i = 0; i < num; ++i)
    {
        int val = input[i*stride];
        largest = (val>largest) ? val : largest;
    }

    for (int i = 0; i < num; ++i)
    {
        float e = expf(input[i*stride] / temperature - largest / temperature);
        sum += e;
        output[i*stride] = e;
    }

    for (int i = 0; i < num; ++i)
    {
        output[i*stride] /= sum;
    }
}

__global__ void softmaxKernel(float *const input, const int num, const int batch, const int batchOff,const int groups,
                               const int groupOff, const float temperature,  const int stride, float *const output)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if( index < batch*groups)
    {

        int _batch = index / groups;
        int _group = index % groups;

        _softmaxKernel(input + _batch*batchOff + _group*groupOff , num, temperature, stride, output + _batch*batchOff + _group*groupOff);
    }
}

void BlasGPU::gpuSoftmax(float *const &input, const int &num, const int &batch, const int &batchOff,const int &groups,
                      const int &groupOff, const float &temperature,  const int &stride, float *const &output)
{
    softmaxKernel<<<Cuda::getGrid(batch*groups), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(input, num, batch, batchOff, groups,
                                                                                                groupOff,temperature,stride,output);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void softMaxCrossEntropyKernel(const int num, float *const pred, float *const truth, float *const delta, float *const error)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(index < num)
    {
        float t     =   truth[index];
        float p     =   pred[index];

        error[index]    =   (t >0 ) ? -logf(p) : 0;
        delta[index]    =   t - p;
    }

}

void BlasGPU::gpuSoftMaxCrossEntropy(const int &num, float *const &pred, float *const &truth, float *const &delta, float *const &error)
{
    softMaxCrossEntropyKernel<<<Cuda::getGrid(num), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(num, pred, truth, delta, error);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void logisticCorssEntropyKernel(const int num, float *const pred, float *const truth, float *const delta, float *const error)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(index < num)
    {
        float t     =   truth[index];
        float p     =   pred[index];

        error[index]    =   -t*logf(p) - (1-t)*logf(1-p);
        delta[index]    =   t - p;
    }

}

void BlasGPU::gpuLogisticCorssEntropy(const int &num, float *const &pred, float *const &truth, float *const &delta, float *const &error)
{
    logisticCorssEntropyKernel<<<Cuda::getGrid(num), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(num, pred, truth, delta, error);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void upSampleKernel(const size_t n,  float *const in, const int width, const int height, const int channel, const int batch,
                             const int stride, const int forward, const float scale, float *const out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < n)
    {
        int outIndex    = i;
        int outW        = i % (width*stride);  

        i = i / (width*stride);
        int outH        = i % (height*stride); 

        i = i / (height*stride);
        int outC        = i % (channel);

        i = i / channel;
        int _batch      = i % (batch); 

        int inW         = outW / stride;
        int inH         = outH / stride;
        int inC         = outC;

        int inIndex     = _batch*width*height*channel + inC*width*height + inH*width + inW;

        if(forward)
        {
            out[outIndex] += scale*in[inIndex];
        }
        else
        {

            atomicAdd(in + inIndex, scale*out[outIndex]);
        }

    }
}

void BlasGPU::gpuUpSample(float *const &in, const int &width, const int &height, const int &channel, const int &batch, const int &stride,
                        const int &forward, const float &scale, float *const &out)
{
    size_t n = width * height * channel * batch * stride * stride;
    upSampleKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, in, width, height, channel, batch, stride,
                                                                                      forward, scale, out);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void addBiasKernel(const int n, float *const output, float *const biases, const int batch, const int num, const int whSize)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if( i < n)
    {
        int f = (i/whSize)%num;
        output[i] += biases[f];
    }
}

void BlasGPU::gpuAddBias(float *const &output, float *const &biases, const int &batch, const int &num, const int &whSize)
{
    const int size = batch*num*whSize;
    addBiasKernel<<<Cuda::getGrid(size), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(size, output, biases, batch, num, whSize);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void scaleBiasKernel(const int n, float *const output, float *const scales, const int batch, const int num, const int whSize)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if( i < n)
    {
        int f = (i/whSize)%num;
        output[i] *= scales[f];
    }
}

void BlasGPU::gpuScaleBias(float *const &output, float *const &scales, const int &batch, const int &num, const int &whSize)
{
    const int size = batch*num*whSize;
    scaleBiasKernel<<<Cuda::getGrid(size), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(size, output, scales, batch, num, whSize);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void fixNanAndInfKernel(float *const input, const size_t size)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if( i < size)
    {
        float val = input[i];
        if(isnan(val) || isinf(val))
        {
            input[i] = 1.f/i;  

        }
    }
}

void BlasGPU::gpuFixNanAndInf(float *const &input, const size_t &size)
{
    fixNanAndInfKernel<<<Cuda::getGrid(size), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(input,size);
    CUDA_CHECK(cudaPeekAtLastError());
}

}

