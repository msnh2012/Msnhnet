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
                                                                   convParams->activation, convParams->actParams, convParams->batchNorm, convParams->bnEps, convParams->useBias,
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
                                                               connectParams->actParams, connectParams->batchNorm, connectParams->bnEps, connectParams->useBias);
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
            layer                           =   new BatchNormLayer(params.batch, params.width, params.height, params.channels, batchNormParams->activation, batchNormParams->eps, batchNormParams->actParams);
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

        if(layer->getMaxOutputNum() >this->_maxOutputNum)
        {
            this->_maxOutputNum  = layer->getMaxOutputNum();
        }

        this->_numWeights    =   this->_numWeights + layer->getNumWeights();
        this->_layerDetail   =   this->_layerDetail.append(layer->getLayerDetail());

        baseLayers.push_back(layer);

        if(layer->getMemReUse()==0)
        {
            layer->mallocMemory();
        }

        if(i == (baseParams.size()-1))
        {
            layer->mallocMemory();
        }

        layer->setIsBranchLayer(true);

        if(i == 0)
        {
            layer->setBranchFirst(true);
        }

        if(i==(baseParams.size()-1))
        {
            layer->setBranchLast(true);
        }
    }

    this->_outHeight         =   params.height;
    this->_outWidth          =   params.width;
    this->_outChannel        =   params.channels;
    this->_outputNum         =   params.inputNums;

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

void ResBlockLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu) 

            {
                this->_output            =   new float[static_cast<size_t>(_outputNum * this->_batch)]();
            }
#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuOutput         =   Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced    =   true;
        }
    }
    this->_memReUse           =   0;
}

void ResBlockLayer::forward(NetworkState &netState)
{
    /* TODO: batch */

    float *layerInput   = nullptr;
    float *layerOutput  = nullptr;

    if(netState.net->layers[this->_layerIndex-1]->getMemReUse() == 1)
    {
        layerInput      = netState.getInput();
    }
    else
    {
        layerInput      = netState.input;
    }

    std::vector<float> inputX{layerInput, layerInput + netState.inputNum};

    netState.input      = inputX.data();

    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        baseLayers[i]->forward(netState);

        if(baseLayers[i]->getMemReUse() == 0) 

        {
            netState.input     =   baseLayers[i]->getOutput();
        }
        netState.inputNum  =   baseLayers[i]->getOutputNum();
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

    Blas::cpuAxpy(netState.inputNum, 1.f, inputX.data(), 1,netState.input, 1);
    Blas::cpuCopy(netState.inputNum, netState.input, 1, layerOutput, 1);

    if(this->_activation == ActivationType::NORM_CHAN)
    {
        Activations::activateArrayNormCh(layerOutput, this->_outputNum, this->_batch, this->_outChannel,
                                         this->_outWidth*this->_outHeight, layerOutput);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        Activations::activateArrayNormChSoftMax(layerOutput, this->_outputNum, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, layerOutput,0);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        Activations::activateArrayNormChSoftMax(layerOutput, this->_outputNum, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, layerOutput,1);
    }
    else if(this->_activation == ActivationType::NONE)
    {

    }
    else
    {
        if(_actParams.size() > 0)
        {
            Activations::activateArray(layerOutput, this->_outputNum, this->_activation, this->supportAvx, _actParams[0]);
        }
        else
        {
            Activations::activateArray(layerOutput, this->_outputNum, this->_activation, this->supportAvx);
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
    float *layerGpuInput   = nullptr;
    float *layerGpuOutput  = nullptr;

    if(netState.net->layers[this->_layerIndex-1]->getMemReUse() == 1)
    {
        layerGpuInput      = netState.getGpuInput();
    }
    else
    {
        layerGpuInput      = netState.input;
    }

    float * inputX      = Cuda::makeCudaArray(layerGpuInput,netState.inputNum,cudaMemcpyKind::cudaMemcpyDeviceToDevice);

    netState.input      = inputX;

    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        baseLayers[i]->forwardGPU(netState);

        if(baseLayers[i]->getMemReUse() == 0) 

        {
            netState.input     =   baseLayers[i]->getGpuOutput();
        }
        netState.inputNum  =   baseLayers[i]->getOutputNum();
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

    BlasGPU::gpuAxpy(netState.inputNum, 1.f, inputX, 1,netState.input, 1);
    BlasGPU::gpuCopy(netState.inputNum, netState.input, 1, layerGpuOutput, 1);

    if(this->_activation == ActivationType::NORM_CHAN)
    {
        ActivationsGPU::gpuActivateArrayNormCh(layerGpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                               this->_outWidth*this->_outHeight, layerGpuOutput);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        ActivationsGPU::gpuActivateArrayNormChSoftMax(layerGpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                      this->_outWidth*this->_outHeight, layerGpuOutput,0);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        ActivationsGPU::gpuActivateArrayNormChSoftMax(layerGpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                      this->_outWidth*this->_outHeight, layerGpuOutput,1);
    }
    else if(this->_activation == ActivationType::NONE)
    {

    }
    else
    {                           

        if(_actParams.size() > 0)
        {
            ActivationsGPU::gpuActivateArray(layerGpuOutput, this->_outputNum*this->_batch, this->_activation, _actParams[0]);
        }
        else
        {
            ActivationsGPU::gpuActivateArray(layerGpuOutput, this->_outputNum*this->_batch, this->_activation);
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
