#include "Msnhnet/layers/MsnhResBlockLayer.h"

namespace Msnhnet
{
ResBlockLayer::ResBlockLayer(const int &batch, NetBuildParams &params, std::vector<BaseParams *> &baseParams, ActivationType &activation,  const std::vector<float> &actParams)
{
    this->_type          =   LayerType::RES_BLOCK;
    this->_layerName     =  "ResBlock        ";
    this->_activation    =   activation;
    this->_actParams     =   actParams;

    this->_batch         =   batch;
    this->_width         =   params.width;
    this->_height        =   params.height;
    this->_channel       =   params.channels;

    BaseLayer *layer    =   nullptr;
    this->_layerDetail.append("================================  ResBlock ================================\n");

    for (size_t i = 0; i < baseParams.size(); ++i)
    {
        if(baseParams[i]->type == LayerType::CONVOLUTIONAL)
        {
            if(params.height ==0 || params.width == 0 || params.channels == 0)
            {
                throw Exception(1, "Layer before convolutional layer must output image", __FILE__, __LINE__, __FUNCTION__);
            }

            ConvParams* convParams      =   reinterpret_cast<ConvParams*>(baseParams[i]);
            layer                       =   new ConvolutionalLayer(params.batch, 1, params.height, params.width, params.channels, convParams->filters,convParams->groups,
                                                                   convParams->kSizeX, convParams->kSizeY, convParams->strideX, convParams->strideY, convParams->dilationX,
                                                                   convParams->dilationY,convParams->paddingX, convParams->paddingY,
                                                                   convParams->activation, convParams->actParams, convParams->batchNorm, convParams->useBias,
                                                                   0,0,0,0,convParams->antialiasing, nullptr, 0,0);

            if(i == 0)
            {
                this->_inputNum = layer->getInputNum();
            }
        }
        else if(baseParams[i]->type == LayerType::CONNECTED)
        {
            ConnectParams *connectParams=   reinterpret_cast<ConnectParams*>(baseParams[i]);
            layer                       =   new ConnectedLayer(params.batch, 1, params.inputNums, connectParams->output, connectParams->activation,
                                                               connectParams->actParams, connectParams->batchNorm);
            if(i == 0)
            {
                this->_inputNum = layer->getInputNum();
            }
        }
        else if(baseParams[i]->type == LayerType::MAXPOOL)
        {
            MaxPoolParams *maxPoolParams=   reinterpret_cast<MaxPoolParams*>(baseParams[i]);
            layer                       =   new MaxPoolLayer(params.batch, params.height, params.width, params.channels, maxPoolParams->kSizeX, maxPoolParams->kSizeY,
                                                             maxPoolParams->strideX, maxPoolParams->strideY, maxPoolParams->paddingX, maxPoolParams->paddingY,
                                                             maxPoolParams->maxPoolDepth, maxPoolParams->outChannels, maxPoolParams->ceilMode, 0);
            if(i == 0)
            {
                this->_inputNum = layer->getInputNum();
            }
        }
        else if(baseParams[i]->type == LayerType::LOCAL_AVGPOOL)
        {
            LocalAvgPoolParams *localAvgPoolParams  =   reinterpret_cast<LocalAvgPoolParams*>(baseParams[i]);
            layer                                   =   new LocalAvgPoolLayer(params.batch, params.height, params.width, params.channels,
                                                                              localAvgPoolParams->kSizeX, localAvgPoolParams->kSizeY, localAvgPoolParams->strideX, localAvgPoolParams->strideY,
                                                                              localAvgPoolParams->paddingX, localAvgPoolParams->paddingY, localAvgPoolParams->ceilMode,0);
            if(i == 0)
            {
                this->_inputNum = layer->getInputNum();
            }
        }
        else if(baseParams[i]->type == LayerType::BATCHNORM)
        {
            BatchNormParams *batchNormParams=   reinterpret_cast<BatchNormParams*>(baseParams[i]);
            layer                           =   new BatchNormLayer(params.batch, params.width, params.height, params.channels, batchNormParams->activation, batchNormParams->actParams);
            if(i == 0)
            {
                this->_inputNum = layer->getInputNum();
            }
        }
        else if(baseParams[i]->type == LayerType::PADDING)
        {
            PaddingParams *paddingParams =   reinterpret_cast<PaddingParams*>(baseParams[i]);
            layer                        =   new PaddingLayer(params.batch, params.height, params.width, params.channels, paddingParams->top,
                                                              paddingParams->down, paddingParams->left, paddingParams->right, paddingParams->paddingVal);
            if(i == 0)
            {
                this->_inputNum = layer->getInputNum();
            }
        }
        else
        {
            throw Exception(1, "layer type is not supported by [ResBlockLayer]", __FILE__, __LINE__, __FUNCTION__);
        }

        params.height       =   layer->getOutHeight();
        params.width        =   layer->getOutWidth();
        params.channels     =   layer->getOutChannel();
        params.inputNums    =   layer->getOutputNum();

        if(layer->getWorkSpaceSize() > this->_workSpaceSize)
        {
            this->_workSpaceSize = layer->getWorkSpaceSize();
        }

        this->_numWeights    =   this->_numWeights + layer->getNumWeights();
        this->_layerDetail   =   this->_layerDetail.append(layer->getLayerDetail());

        baseLayers.push_back(layer);
    }

    this->_outHeight         =   params.height;
    this->_outWidth          =   params.width;
    this->_outChannel        =   params.channels;
    this->_outputNum         =   params.inputNums;

    if(!BaseLayer::isPreviewMode)
    {
        this->_output            =   new float[static_cast<size_t>(_outputNum * this->_batch)]();
#ifdef USE_GPU
        this->_gpuOutput         = Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
#endif
    }
    this->_layerDetail.append("========================================================================\n");
}

void ResBlockLayer::loadAllWeigths(std::vector<float> &weights)
{

    if(weights.size() != this->_numWeights)
    {
        throw Exception(1,"ResBlock weights load err. needed : " + std::to_string(this->_numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__, __FUNCTION__);
    }

    size_t ptr = 0;
    std::vector<float>::const_iterator first = weights.begin();

    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        if(baseLayers[i]->type() == LayerType::CONVOLUTIONAL || baseLayers[i]->type() == LayerType::CONNECTED || baseLayers[i]->type() == LayerType::BATCHNORM)
        {
            size_t nums = baseLayers[i]->getNumWeights();

            std::vector<float> weights(first + static_cast<long long>(ptr), first + static_cast<long long>(ptr + nums));

            baseLayers[i]->loadAllWeigths(weights);

            ptr         =   ptr + nums;
        }
    }
}

void ResBlockLayer::forward(NetworkState &netState)
{
    /* TODO: batch */
    std::vector<float> inputX{netState.input, netState.input + netState.inputNum};

    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        baseLayers[i]->forward(netState);

        netState.input     =   baseLayers[i]->getOutput();
        netState.inputNum  =   baseLayers[i]->getOutputNum();
    }

    Blas::cpuAxpy(netState.inputNum, 1.f, inputX.data(), 1,netState.input, 1);
    Blas::cpuCopy(netState.inputNum, netState.input, 1, this->_output, 1);

    if(this->_activation == ActivationType::NORM_CHAN)
    {
        Activations::activateArrayNormCh(this->_output, this->_outputNum, this->_batch, this->_outChannel,
                                         this->_outWidth*this->_outHeight, this->_output);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        Activations::activateArrayNormChSoftMax(this->_output, this->_outputNum, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, this->_output,0);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        Activations::activateArrayNormChSoftMax(this->_output, this->_outputNum, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, this->_output,1);
    }
    else if(this->_activation == ActivationType::NONE)
    {

    }
    else
    {
        if(_actParams.size() > 0)
        {
            Activations::activateArray(this->_output, this->_outputNum, this->_activation, this->supportAvx, _actParams[0]);
        }
        else
        {
            Activations::activateArray(this->_output, this->_outputNum, this->_activation, this->supportAvx);
        }
    }

    this->_forwardTime = 0;

    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        this->_forwardTime += baseLayers[i]->getForwardTime();
    }

}

#ifdef USE_GPU
void ResBlockLayer::forwardGPU(NetworkState &netState)
{
    float * inputX      = Cuda::makeCudaArray(netState.input,netState.inputNum,cudaMemcpyKind::cudaMemcpyDefault);

    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        baseLayers[i]->forwardGPU(netState);

        netState.input     =   baseLayers[i]->getGpuOutput();
        netState.inputNum  =   baseLayers[i]->getOutputNum();
    }

    BlasGPU::gpuAxpy(netState.inputNum, 1.f, inputX, 1,netState.input, 1);
    BlasGPU::gpuCopy(netState.inputNum, netState.input, 1, this->_gpuOutput, 1);

    if(this->_activation == ActivationType::NORM_CHAN)
    {
        ActivationsGPU::gpuActivateArrayNormCh(this->_gpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                            this->_outWidth*this->_outHeight, this->_gpuOutput);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        ActivationsGPU::gpuActivateArrayNormChSoftMax(this->_gpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, this->_gpuOutput,0);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        ActivationsGPU::gpuActivateArrayNormChSoftMax(this->_gpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, this->_gpuOutput,1);
    }
    else if(this->_activation == ActivationType::NONE)
    {

    }
    else
    {                           

        if(_actParams.size() > 0)
        {
            ActivationsGPU::gpuActivateArray(this->_gpuOutput, this->_outputNum*this->_batch, this->_activation, _actParams[0]);
        }
        else
        {
            ActivationsGPU::gpuActivateArray(this->_gpuOutput, this->_outputNum*this->_batch, this->_activation);
        }
    }

    this->_forwardTime = 0;

    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        this->_forwardTime += baseLayers[i]->getForwardTime();
    }

    Cuda::freeCuda(inputX);
}
#endif

ResBlockLayer::~ResBlockLayer()
{
    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        if(baseLayers[i]!=nullptr)
        {
            if(baseLayers[i]->type() == LayerType::CONVOLUTIONAL)
            {
                delete reinterpret_cast<ConvolutionalLayer*>(baseLayers[i]);
            }
            else if(baseLayers[i]->type() == LayerType::MAXPOOL)
            {
                delete reinterpret_cast<MaxPoolLayer*>(baseLayers[i]);
            }
            else if(baseLayers[i]->type() == LayerType::CONNECTED)
            {
                delete reinterpret_cast<ConnectedLayer*>(baseLayers[i]);
            }
            else if(baseLayers[i]->type() == LayerType::BATCHNORM)
            {
                delete reinterpret_cast<BatchNormLayer*>(baseLayers[i]);
            }
            else if(baseLayers[i]->type() == LayerType::LOCAL_AVGPOOL)
            {
                delete reinterpret_cast<LocalAvgPoolLayer*>(baseLayers[i]);
            }
            else if(baseLayers[i]->type() == LayerType::PADDING)
            {
                delete reinterpret_cast<PaddingLayer*>(baseLayers[i]);
            }

            baseLayers[i] = nullptr;
        }

        if(i == (baseLayers.size()-1))
        {
            baseLayers.clear();
        }
    }
}
}
