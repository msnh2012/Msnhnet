#include "Msnhnet/layers/cuda/MsnhActivationsGPU.h"

namespace Msnhnet
{
__device__ float logisticActivateKernel(const float x)
{
    return 1.f/(1.f + expf(-x));
}

__device__ float loggyActivateKernel(const float x)
{
    return 2.f/(1.f + expf(-x)) - 1.f;
}

__device__ float reluActivateKernel(const float x)
{
    return x*(x>0);
}

__device__ float relu6ActivateKernel(const float x)
{
    return (x>0?x:0)>6?6:(x>0?x:0);
}

__device__ float eluActivateKernel(const float x)
{
    return ((x >= 0)*x + (x < 0)*(expf(x)-1.f));
}

__device__ float seluActivateKernel(const float x)
{
    return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x) - 1);
}

__device__ float relieActivateKernel(const float x)
{
    return (x>0) ? x : .01f*x;
}

__device__ float rampActivateKernel(const float x)
{
    return x*(x>0) + .1f*x;
}

__device__ float leakyActivateKernel(const float x, const float param = 0.1f)
{
    return (x>0) ? x : param*x;
}

__device__ float tanhActivateKernel(const float x)
{
    return ((expf(2*x)-1)/(expf(2*x)+1));
}

__device__ float stairActivateKernel(const float x)
{
    int n = static_cast<int>(floor(x));
    if (n%2 == 0)
    {
        return (floorf(x/2.f));
    }
    else
    {
        return static_cast<float>((x - n) + floorf(x/2.f));
    }
}

__device__ float hardtanActivateKernel(const float x)
{
    if (x < -1)
    {
        return -1;
    }
    if (x > 1)
    {
        return 1;
    }
    return x;
}

__device__ float softplusActivateKernel(const float x, const float threshold)
{
    if (x > threshold)
    {
        return x;
    }
    else if (x < -threshold)
    {
        return expf(x);
    }
    return logf(expf(x) + 1);
}

__device__ float plseActivateKernel(const float x)
{
    if(x < -4)
    {
        return .01f * (x + 4);
    }
    if(x > 4)
    {
        return .01f * (x - 4) + 1;
    }
    return .125f*x + .5f;
}

__device__ float lhtanActivateKernel(const float x)
{
    if(x < 0.0f)
    {
        return .001f*x;
    }
    if(x > 1.0f)
    {
        return .001f*(x-1) + 1;
    }
    return x;
}

__device__ float mishActivateKernel(const float x)
{
    const float mishThreshHold = 20.f;
    return x*tanhf(softplusActivateKernel(x, mishThreshHold));
}

__device__ float swishActivateKernel(const float x)
{
    return x*logisticActivateKernel(x);
}
__device__ float activateKernel(const float x, const ActivationType actType, const float params)
{
    switch (actType)
    {
    case LOGISTIC:
        return logisticActivateKernel(x);
    case LOGGY:
        return loggyActivateKernel(x);
    case RELU:
        return reluActivateKernel(x);
    case RELU6:
        return relu6ActivateKernel(x);
    case ELU:
        return eluActivateKernel(x);
    case SELU:
        return seluActivateKernel(x);
    case RELIE:
        return relieActivateKernel(x);
    case RAMP:
        return rampActivateKernel(x);
    case LEAKY:
        return leakyActivateKernel(x, params);
    case TANH:
        return tanhActivateKernel(x);
    case PLSE:
        return plseActivateKernel(x);
    case STAIR:
        return stairActivateKernel(x);
    case HARDTAN:
        return hardtanActivateKernel(x);
    case LHTAN:
        return lhtanActivateKernel(x);
    case SOFT_PLUS:
        return softplusActivateKernel(x, params);
    case MISH:
        return mishActivateKernel(x);
    case SWISH:
        return swishActivateKernel(x);
    case NONE:
        return x;
    default:
        return 0;
    }
}

__global__ void activateArrayKernel(float *const x, const int numX, const ActivationType  actType,  const float  param)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < numX)
    {
        x[i] = activateKernel(x[i],actType,param);
    }
}

void ActivationsGPU::gpuActivateArray(float *const &gpuX, const int &numX, const ActivationType &actType, const float &param)
{
    activateArrayKernel<<<Cuda::getGrid(numX), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(gpuX,numX,actType,param);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void activateArrayNormChKernel(float *const gpuX, const int numX, const int batch, const int channels, const int whStep, float *const gpuOutput)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    int whIndex = i%whStep;
    int b       = whStep;

    const float eps = 0.0001;

    if(i < numX)
    {
        float sum = eps;

        for (int c = 0; c < channels; ++c)
        {
            float val = gpuX[whIndex + c*whStep + b*whStep*channels];
            if(val > 0)
            {
                sum += val;
            }
        }

        for (int c = 0; c < channels; ++c)
        {
            float val = gpuX[whIndex + c*whStep + b*whStep*channels];
            if(val > 0)
            {
                val = val/sum;
            }
            else
            {
                val = 0;
            }

            gpuOutput[whIndex + c*whStep + b*whStep*channels] = val;
        }
    }
}

void ActivationsGPU::gpuActivateArrayNormCh(float *const &gpuX, const int &numX, const int &batch, const int &channels, const int &whStep, float *const &gpuOutput)
{
    activateArrayNormChKernel<<<Cuda::getGrid(numX), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(gpuX,numX,batch,channels,whStep,gpuOutput);
    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void activateArrayNormChSoftMaxKernel(float *const &gpuX, const int &numX, const int &batch, const int &channels, const int &whStep, float *const &gpuOutput, const int &useMaxVal)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;

    int whIndex = i%whStep;
    int b       = whStep;
    const float eps = 0.0001;

    if(i < numX)
    {
        float sum       = eps;
        float maxVal    = -FLT_MAX;

        if(useMaxVal)
        {
            for (int c = 0; c < channels; ++c)
            {
                float val = gpuX[whIndex + c*whStep + b*whStep*channels];
                if(val > maxVal || c == 0)
                {
                    maxVal = val;
                }
            }
        }
        else
        {
            maxVal = 0;
        }

        for (int c = 0; c < channels; ++c)
        {
            float val = gpuX[whIndex + c*whStep + b*whStep*channels];
            sum += expf(val - maxVal);
        }

        for (int c = 0; c < channels; ++c)
        {
            float val = gpuX[whIndex + c*whStep + b*whStep*channels];
            val = expf(val - maxVal)/sum;
            gpuOutput[whIndex + c*whStep + b*whStep*channels] = val;
        }
    }
}

void ActivationsGPU::gpuActivateArrayNormChSoftMax(float *const &gpuX, const int &numX, const int &batch, const int &channels, const int &whStep, float *const &gpuOutput, const int &useMaxVal)
{
    activateArrayNormChSoftMaxKernel<<<Cuda::getGrid(numX), Cuda::blockThread, 0, Cuda::getCudaStream()>>>(gpuX,numX,batch,channels,whStep,gpuOutput,useMaxVal);
    CUDA_CHECK(cudaPeekAtLastError());
}

}
