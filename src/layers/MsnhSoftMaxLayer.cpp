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

    if(!BaseLayer::isPreviewMode)
    {
        this->_output        =   new float[static_cast<size_t>(this->_inputNum * this->_batch)]();
#ifdef USE_GPU
        this->_gpuOutput         = Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
#endif
    }

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
    TimeUtil::startRecord();

    Blas::cpuSoftmax(netState.input, this->_inputNum/this->_groups, this->_batch, this->_inputNum,
                     this->_groups, this->_inputNum/this->_groups, this->_temperature, 1, this->_output, BaseLayer::supportAvx);

    this->_forwardTime =   TimeUtil::getElapsedTime();
}

#ifdef USE_GPU
void SoftMaxLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();
#ifdef USE_CUDNN
    if(!onlyUseCuda)
    {
        float a = 1.f;
        float b = 0;
        CUDNN_CHECK(cudnnSoftmaxForward(Cuda::getCudnnHandle(), CUDNN_SOFTMAX_FAST,
                                        CUDNN_SOFTMAX_MODE_CHANNEL,
                                        &a,
                                        _inputDesc, netState.input,
                                        &b,
                                        _outputDesc, this->_gpuOutput));
    }
    else
    {
        BlasGPU::gpuSoftmax(netState.input, this->_inputNum/this->_groups, this->_batch, this->_inputNum,
                         this->_groups, this->_inputNum/this->_groups, this->_temperature, 1, this->_gpuOutput);
    }
#else

    BlasGPU::gpuSoftmax(netState.input, this->_inputNum/this->_groups, this->_batch, this->_inputNum,
                     this->_groups, this->_inputNum/this->_groups, this->_temperature, 1, this->_gpuOutput);
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
