#include "Msnhnet/net/MsnhNetwork.h"
namespace Msnhnet
{

NetworkState::~NetworkState()
{
    releaseArr(workspace);
    releaseArr(memPool1);
    releaseArr(memPool2);

#ifdef USE_GPU
    Cuda::freeCuda(gpuMemPool1);
    Cuda::freeCuda(gpuMemPool2);
    Cuda::freeCuda(gpuWorkspace);
    if(BaseLayer::useFp16)
    {
        Cuda::freeCuda(gpuInputFp16);
    }
#endif
}

uint8_t NetworkState::getInputWorkSpace() const
{
    return _inputWorkSpace;
}

uint8_t NetworkState::getOutputWorkSpace() const
{
    return _outputWorkSpace;
}

float *NetworkState::getInput() const
{
    if(_inputWorkSpace == 0)
    {
        return memPool1;
    }
    else
    {
        return memPool2;
    }
}

float *NetworkState::getOutput() const
{
    if(_outputWorkSpace == 0)
    {
        return memPool1;
    }
    else
    {
        return memPool2;
    }
}

#ifdef USE_GPU
float *NetworkState::getGpuInput() const
{
    if(_gpuInputWorkSpace == 0)
    {
        return gpuMemPool1;
    }
    else
    {
        return gpuMemPool2;
    }
}

float *NetworkState::getGpuOutput() const
{
    if(_gpuOutputWorkSpace == 0)
    {
        return gpuMemPool1;
    }
    else
    {
        return gpuMemPool2;
    }
}

uint8_t NetworkState::getGpuInputWorkSpace() const
{
    return _gpuInputWorkSpace;
}

uint8_t NetworkState::getGpuOutputWorkSpace() const
{
    return _gpuOutputWorkSpace;
}
#endif

}
