#include "Msnhnet/layers/MsnhSoftMaxLayer.h"

namespace Msnhnet
{
SoftMaxLayer::SoftMaxLayer(const int &batch, const int &inputNum, const int &groups, const float &temperature)
{
    this->_type          =   LayerType::SOFTMAX;
    this->_layerName     =   "SoftMax         ";

    this->_batch         =   batch;
    this->_groups        =   groups;
    this->_temperature   =   temperature;
    this->_inputNum      =   inputNum;
    this->_outputNum     =   inputNum;

    this->_maxOutputNum  = this->_batch*this->_outputNum;

#ifdef USE_GPU
#ifdef USE_CUDNN

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->_inputDesc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->_inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->_batch, this->_outputNum,1,1));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->_outputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->_outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->_batch, this->_outputNum,1,1));

#endif
#endif

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "softmax                                        %4d\n", inputNum);
#else
    sprintf(msg, "softmax                                        %4d\n", inputNum);
#endif
    this->_layerDetail   = msg;
}

SoftMaxLayer::~SoftMaxLayer()
{
#ifdef USE_GPU
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_outputDesc));
#endif
#endif
}

void SoftMaxLayer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();

    float* layerInput   = netState.getInput();
    float* layerOutput  = nullptr;

    if(this->_layerIndex == 0) 

    {
        layerInput      = netState.input;
    }
    else 

    {
        if(netState.net->layers[this->_layerIndex - 1]->getMemReUse() == 0)

        {
            layerInput  = netState.input;
        }
    }

    if(this->_memReUse==1) 

    {
        layerOutput     = netState.getOutput(); 

        netState.shuffleInOut();

    }
    else

    {
        layerOutput     = this->_output;
    }

    Blas::cpuSoftmax(layerInput, this->_inputNum/this->_groups, this->_batch, this->_inputNum,
                     this->_groups, this->_inputNum/this->_groups, this->_temperature, 1, layerOutput, BaseLayer::supportAvx);

    this->_forwardTime =   TimeUtil::getElapsedTime(st);
}

void SoftMaxLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu) 

            {

                this->_output         = MemoryManager::effcientNew<float>(static_cast<size_t>(this->_outputNum * this->_batch));
            }
#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuOutput     =   Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }
    this->_memReUse         =  0;
}

#ifdef USE_GPU
void SoftMaxLayer::forwardGPU(NetworkState &netState)
{

    float* layerGpuInput   = netState.getGpuInput();
    float* layerGpuOutput  = nullptr;

    if(this->_layerIndex == 0) 

    {
        layerGpuInput      = netState.input;
    }
    else 

    {
        if(netState.net->layers[this->_layerIndex - 1]->getMemReUse() == 0)

        {
            layerGpuInput  = netState.input;
        }
    }

    if(this->_memReUse==1) 

    {
        layerGpuOutput     = netState.getGpuOutput(); 

        netState.shuffleGpuInOut();

    }
    else

    {
        layerGpuOutput     = this->_gpuOutput;
    }

    this->recordCudaStart();

#ifdef USE_CUDNN
    if(!onlyUseCuda)
    {
        float a = 1.f;
        float b = 0;
        CUDNN_CHECK(cudnnSoftmaxForward(Cuda::getCudnnHandle(), CUDNN_SOFTMAX_FAST,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &a,
                                        _inputDesc, layerGpuInput,
                                        &b,
                                        _outputDesc, layerGpuOutput));
    }
    else
    {
        BlasGPU::gpuSoftmax(layerGpuInput, this->_inputNum/this->_groups, this->_batch, this->_inputNum,
                            this->_groups, this->_inputNum/this->_groups, this->_temperature, 1, layerGpuOutput);
    }
#else

    BlasGPU::gpuSoftmax(layerGpuInput, this->_inputNum/this->_groups, this->_batch, this->_inputNum,
                        this->_groups, this->_inputNum/this->_groups, this->_temperature, 1, layerGpuOutput);
#endif
    this->recordCudaStop();
}
#endif

int SoftMaxLayer::getGroups() const
{
    return _groups;
}

float SoftMaxLayer::getTemperature() const
{
    return _temperature;
}
}
