#include "Msnhnet/core/cuda/MsnhBlasGPU.h"
namespace Msnhnet
{

__global__ void copySimpleKernel(const int size,  float * const src, float * const dst)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index < size)
        dst[index] = src[index];
}

void BlasGPU::gpuSimpleCopy(const int &n,  float * const &src, float * const &dst)
{

    copySimpleKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, src, dst);
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
__global__ void fillKernel(const int n, const float alpha, float *const x, const int step)
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
            float tmp01 = (alpha==0?0.000001:alpha);
            out[i*stepOut] = x[i*stepX] / tmp01 ;
            break;
        case ARITH_DIV_INV:
            float tmp02 = (x[i*stepX]==0?0.000001:x[i*stepX]);
            out[i*stepOut] = alpha / tmp02;
            break;
        }
    }
}

void BlasGPU::gpuArithmetic(const Arithmetic &type, const int &n, float * const &x, const int &stepX, const float &alpha, float *out, const int &stepOut)
{
    arithmeticConstKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(type, n, x, stepX, alpha, out, stepOut);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void  scientificKernel(const Scientific type, const int n, float * const x, const int stepX, const float alpha, float *out, const int stepOut)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i<n)
    {
        switch (type)
        {
        case SCI_ABS:
            out[i*stepOut] = abs(x[i*stepX]);
            break;
        case SCI_ACOS:
            out[i*stepOut] = acosf(x[i*stepX]);
            break;
        case SCI_ASIN:
            out[i*stepOut] = asinf(x[i*stepX]);
            break;
        case SCI_ATAN:
            out[i*stepOut] = atanf(x[i*stepX]);
            break;
        case SCI_COS:
            out[i*stepOut] = cosf(x[i*stepX]);
            break;
        case SCI_COSH:
            out[i*stepOut] = coshf(x[i*stepX]);
            break;
        case SCI_SIN:
            out[i*stepOut] = sinf(x[i*stepX]);
            break;
        case SCI_SINH:
            out[i*stepOut] = sinhf(x[i*stepX]);
            break;
        case SCI_TAN:
            out[i*stepOut] = tanf(x[i*stepX]);
            break;
        case SCI_TANH:
            out[i*stepOut] = tanhf(x[i*stepX]);
            break;
        case SCI_EXP:
            out[i*stepOut] = expf(x[i*stepX]);
            break;
        case SCI_POW:
            out[i*stepOut] = powf(x[i*stepX],alpha);
            break;
        case SCI_LOG:
            out[i*stepOut] = logf(x[i*stepX]);
            break;
        case SCI_LOG10:
            out[i*stepOut] = log10f(x[i*stepX]);
            break;
        case SCI_SQRT:
            out[i*stepOut] = sqrtf(x[i*stepX]);
            break;
        }
    }
}

void BlasGPU::gpuScientific(const Scientific &type, const int &n, float *const &x, const int &stepX, const float alpha, float *out, const int &stepOut)
{
    scientificKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(type, n, x, stepX, alpha, out, stepOut);
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

__global__ void fastSumKernel(const int n, const float * in, float * sum, const ReductionType type)
{

    __shared__ float buffer[CUDA_THREADS];

    buffer[threadIdx.x]=0;

    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        buffer[threadIdx.x] += in[i];
    }

    __syncthreads(); 

    for (int s = blockDim.x/2; s >0 ; s>>=1)
    {
        if (threadIdx.x < s)
            buffer[threadIdx.x] += buffer[threadIdx.x+s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        if(type == ReductionType::REDUCTION_SUM)
        {
            *sum = buffer[0];
        }
        else
        {
            *sum = buffer[0]/n;
        }
    }
}

__global__ void sumKernelEx(const int n, const int axis, const int batch, const int channel, const int width, const int height, float *const input, float *output, const ReductionType type)
{
    int index   = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(index < n)
    {
        int w = index % width;
        index = index / width;
        int h = index % height;
        index = index / height;
        int c = index % channel;
        index = index / channel;
        int b = index;

        if(axis == 0)
        {

            atomicAdd(&output[b*height*width + h*width + w], input[b*channel*height*width + c*height*width + h*width + w]);
        }
        else if(axis == 1)
        {

            atomicAdd(&output[b*channel*width + c*width + w], input[b*channel*height*width + c*height*width + h*width + w]);
        }
        else if(axis == 2)
        {

            atomicAdd(&output[b*channel*height + c*height + h], input[b*channel*height*width + c*height*width + h*width + w]);
        }
        __syncthreads();
    }
}

void BlasGPU::gpuFastSum(const int &axis, const int &batch, const int &channel, const int &width, const int &height, float *const &input, float *output, const ReductionType &reductionType)
{
    if(axis == -1)
    {
        int inN = channel * width * height;
        for (int i = 0; i < batch; ++i)
        {
            fastSumKernel<<<1,Cuda::blockThread,0,Cuda::getCudaStream()>>>(inN, input+inN*i, &(output[i]),reductionType);
        }
        CUDA_CHECK(cudaPeekAtLastError());
    }
    else
    {
        int inN = channel * width * height * batch;
        sumKernelEx<<<Cuda::getGrid(inN),Cuda::blockThread,0,Cuda::getCudaStream()>>>(inN, axis, batch, channel, width, height, input, output, reductionType);
        CUDA_CHECK(cudaPeekAtLastError());
        if(axis == 0)
        {
            int num =  width * height * batch;
            gpuScale(num, 1.f/channel, output, 1 );
        }
        else if(axis == 1)
        {
            int num =  channel * width  * batch;
            gpuScale(num, 1.f/height, output, 1 );
        }
        else if(axis == 2)
        {
            int num =  channel * height  * batch;
            gpuScale(num, 1.f/width, output, 1 );
        }
    }
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
                               const int strideX, const int strideY, const float scale, float *const out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(i < n)
    {
        int outIndex    = i;
        int outW        = i % (width*strideX);  

        i = i / (width*strideX);
        int outH        = i % (height*strideY); 

        i = i / (height*strideY);
        int outC        = i % (channel);

        i = i / channel;
        int _batch      = i % (batch); 

        int inW         = outW / strideX;
        int inH         = outH / strideY;
        int inC         = outC;

        int inIndex     = _batch*width*height*channel + inC*width*height + inH*width + inW;

        out[outIndex] += scale*in[inIndex];

    }
}

void BlasGPU::gpuUpSample(float *const &in, const int &width, const int &height, const int &channel, const int &batch, const int &strideX,
                          const int &strideY, const float &scale, float *const &out)
{
    size_t n = width * height * channel * batch * strideX * strideY;
    upSampleKernel<<<Cuda::getGrid(n), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(n, in, width, height, channel, batch, strideX,
                                                                                      strideY, scale, out);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void bilinearResizeKernel(const int n, float *const in, const int width, const int height, const int channel, const int batch,
                                     const size_t inSize, const int outSize, const float rHeight, const float rWidth,
                                     const int outWidth, const int outHeight, const int alignCorners, float *const out)
{
    size_t idx = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    if(idx < n)
    {

        int tmpC    = channel*batch;
        int c       = idx % tmpC;
        idx         = idx / tmpC;
        int i       = idx % outSize;

        const int h2 = i / outWidth;
        const int w2 = i % outWidth;
        float h1r = 0;
        float w1r = 0;

        if(alignCorners==0)
        {
            const float inIdxH =  rHeight*(h2+0.5f)-0.5f;
            h1r = inIdxH<0?0:inIdxH;

            const float inIdxW =  rWidth*(w2+0.5f)-0.5f;
            w1r = inIdxW<0?0:inIdxW;
        }
        else
        {
            h1r = rHeight*h2;
            w1r = rWidth *w2;
        }

        const int h1 = static_cast<int>(h1r);
        const int w1 = static_cast<int>(w1r);

        const int h1p = (h1 < (height - 1))?1:0;
        const int w1p = (w1 < (width  - 1))?1:0;

        const float h1Lamd =  h1r - h1;
        const float h0Lamd =  1.0f - h1Lamd;

        const float w1Lamd =  w1r - w1;
        const float w0Lamd =  1.0f - w1Lamd;

        const float *inPtr = in  + h1*width + w1;
        float *outPtr      = out + i;

        const float* inTmp = inPtr + c*inSize;
        *(outPtr + c*outSize) = h0Lamd * (w0Lamd*(*inTmp) + w1Lamd*(*(inTmp + w1p)))
                               +h1Lamd * (w0Lamd*(*(inTmp + h1p*width))
                               +w1Lamd * (*(inTmp + h1p*width + w1p)));

    }
}

void BlasGPU::gpuBilinearResize(float *const &in, const int &width, const int &height, const int &channel, const int &batch, const int &outWidth,
                                  const int &outHeight, const int &alignCorners, float *const &out)
{
    if(height<1 || outHeight<1 || width <1 || outWidth <1)
    {
        throw Exception(1,"w*x and outw*outx must > 1",__FILE__, __LINE__, __FUNCTION__);
    }

    const float rHeight = (alignCorners==0)?(1.0f*height/outHeight):(1.0f*(height-1)/(outHeight-1));
    const float rWidth  =  (alignCorners==0)?(1.0f*width/outWidth):(1.0f*(width-1)/(outWidth-1));

    const size_t inSize  = width*height;
    const size_t outSize = outWidth*outHeight;

    const size_t num     = outSize*channel*batch;

    bilinearResizeKernel<<<Cuda::getGrid(num), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(num, in, width, height, channel, batch, inSize, outSize, rHeight, rWidth, outWidth, outHeight, alignCorners, out);
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

