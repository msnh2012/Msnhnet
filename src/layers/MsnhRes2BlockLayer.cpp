#include "Msnhnet/layers/MsnhRes2BlockLayer.h"

namespace Msnhnet
{
Res2BlockLayer::Res2BlockLayer(const int &batch, NetBuildParams &params, std::vector<BaseParams *> &baseParams, std::vector<BaseParams *> &branchParams, ActivationType &activation,  const std::vector<float> &actParams)
{
    this->type          =   LayerType::RES_2_BLOCK;
    this->layerName     =   "Res2Block       ";
    this->activation    =   activation;
    this->actParams     =   actParams;

    this->batch         =   batch;
    this->width         =   params.width;
    this->height        =   params.height;
    this->channel       =   params.channels;

    BaseLayer *layer    =   nullptr;

    NetBuildParams  branchBuildParams = params;

    this->layerDetail.append("================================ Res2Block ================================\n");

    for (size_t i = 0; i < baseParams.size(); ++i)
    {
        if(baseParams[i]->type == LayerType::CONVOLUTIONAL)
        {
            if(params.height ==0 || params.width == 0 || params.channels == 0)
            {
                throw Exception(1, "Layer before convolutional layer must output image", __FILE__, __LINE__);
            }

            ConvParams* convParams      =   reinterpret_cast<ConvParams*>(baseParams[i]);
            layer                       =   new ConvolutionalLayer(params.batch, 1, params.height, params.width, params.channels, convParams->filters,convParams->groups,
                                                                   convParams->kSizeX, convParams->kSizeY, convParams->strideX, convParams->strideY, convParams->dilationX,
                                                                   convParams->dilationY,convParams->paddingX, convParams->paddingY,
                                                                   convParams->activation, convParams->actParams, convParams->batchNorm, convParams->useBias,
                                                                   0,0,0,0,convParams->antialiasing, nullptr, 0,0);

            if(i == 0)
            {
                this->inputNum = layer->inputNum;
            }
        }
        else if(baseParams[i]->type == LayerType::CONNECTED)
        {
            ConnectParams *connectParams=   reinterpret_cast<ConnectParams*>(baseParams[i]);
            layer                       =   new ConnectedLayer(params.batch, 1, params.inputNums, connectParams->output, connectParams->activation,
                                                               connectParams->actParams, connectParams->batchNorm);
            if(i == 0)
            {
                this->inputNum = layer->inputNum;
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
                this->inputNum = layer->inputNum;
            }
        }
        else if(baseParams[i]->type == LayerType::LOCAL_AVGPOOL)
        {
            LocalAvgPoolParams *localAvgPoolParams  =   reinterpret_cast<LocalAvgPoolParams*>(baseParams[i]);
            layer                                   =   new LocalAvgPoolLayer(params.batch, params.height, params.width, params.channels,
                                                                              localAvgPoolParams->kSizeX, localAvgPoolParams->kSizeY, localAvgPoolParams->strideX, localAvgPoolParams->strideY,
                                                                              localAvgPoolParams->paddingX, localAvgPoolParams->paddingY, localAvgPoolParams->ceilMode, 0);
            if(i == 0)
            {
                this->inputNum = layer->inputNum;
            }
        }
        else if(baseParams[i]->type == LayerType::BATCHNORM)
        {
            BatchNormParams *batchNormParams=   reinterpret_cast<BatchNormParams*>(baseParams[i]);
            layer                           =   new BatchNormLayer(params.batch, params.width, params.height, params.channels, batchNormParams->activation, batchNormParams->actParams);
            if(i == 0)
            {
                this->inputNum = layer->inputNum;
            }
        }
        else if(baseParams[i]->type == LayerType::PADDING)
        {
            PaddingParams *paddingParams =   reinterpret_cast<PaddingParams*>(baseParams[i]);
            layer                        =   new PaddingLayer(params.batch, params.height, params.width, params.channels, paddingParams->top,
                                                              paddingParams->down, paddingParams->left, paddingParams->right, paddingParams->paddingVal);
            if(i == 0)
            {
                this->inputNum = layer->inputNum;
            }
        }
        else
        {
            throw Exception(1, "layer type is not supported by [Res2BlockLayer]", __FILE__, __LINE__);
        }

        params.height       =   layer->outHeight;
        params.width        =   layer->outWidth;
        params.channels     =   layer->outChannel;
        params.inputNums    =   layer->outputNum;

        if(layer->workSpaceSize > this->workSpaceSize)
        {
            this->workSpaceSize = layer->workSpaceSize;
        }

        this->numWeights    =   this->numWeights + layer->numWeights;
        this->layerDetail   =   this->layerDetail.append(layer->layerDetail);

        baseLayers.push_back(layer);
    }

    for (size_t i = 0; i < branchParams.size(); ++i)
    {
        if(branchParams[i]->type == LayerType::CONVOLUTIONAL)
        {
            if(branchBuildParams.height ==0 || branchBuildParams.width == 0 || branchBuildParams.channels == 0)
            {
                throw Exception(1, "Layer before convolutional layer must output image", __FILE__, __LINE__);
            }

            ConvParams* convParams      =   reinterpret_cast<ConvParams*>(branchParams[i]);
            layer                       =   new ConvolutionalLayer(branchBuildParams.batch, 1, branchBuildParams.height, branchBuildParams.width, branchBuildParams.channels,
                                                                   convParams->filters,convParams->groups,convParams->kSizeX, convParams->kSizeY, convParams->strideX, convParams->strideY,
                                                                   convParams->dilationX,convParams->dilationY,convParams->paddingX, convParams->paddingY, convParams->activation,
                                                                   convParams->actParams, convParams->batchNorm, convParams->useBias,
                                                                   0,0,0,0,convParams->antialiasing, nullptr, 0,0);

        }
        else if(branchParams[i]->type == LayerType::CONNECTED)
        {
            ConnectParams *connectParams=   reinterpret_cast<ConnectParams*>(branchParams[i]);
            layer                       =   new ConnectedLayer(branchBuildParams.batch, 1, branchBuildParams.inputNums, connectParams->output, connectParams->activation, connectParams->actParams,
                                                               connectParams->batchNorm);
        }
        else if(branchParams[i]->type == LayerType::MAXPOOL)
        {
            MaxPoolParams *maxPoolParams=   reinterpret_cast<MaxPoolParams*>(branchParams[i]);
            layer                       =   new MaxPoolLayer(branchBuildParams.batch, branchBuildParams.height, branchBuildParams.width, branchBuildParams.channels, maxPoolParams->kSizeX, maxPoolParams->kSizeY,
                                                             maxPoolParams->strideX, maxPoolParams->strideY, maxPoolParams->paddingX, maxPoolParams->paddingY,
                                                             maxPoolParams->maxPoolDepth, maxPoolParams->outChannels, maxPoolParams->ceilMode, 0);
        }
        else if(branchParams[i]->type == LayerType::LOCAL_AVGPOOL)
        {
            LocalAvgPoolParams *localAvgPoolParams  =   reinterpret_cast<LocalAvgPoolParams*>(branchParams[i]);
            layer                                   =   new LocalAvgPoolLayer(branchBuildParams.batch, branchBuildParams.height, branchBuildParams.width, branchBuildParams.channels,
                                                                              localAvgPoolParams->kSizeX, localAvgPoolParams->kSizeY, localAvgPoolParams->strideX, localAvgPoolParams->strideY,
                                                                              localAvgPoolParams->paddingX, localAvgPoolParams->paddingY, localAvgPoolParams->ceilMode, 0);
        }
        else if(branchParams[i]->type == LayerType::BATCHNORM)
        {
            BatchNormParams *batchNormParams=   reinterpret_cast<BatchNormParams*>(branchParams[i]);
            layer                           =   new BatchNormLayer(branchBuildParams.batch, branchBuildParams.width, branchBuildParams.height, branchBuildParams.channels, batchNormParams->activation, batchNormParams->actParams);
        }
        else if(branchParams[i]->type == LayerType::PADDING)
        {
            PaddingParams *paddingParams =   reinterpret_cast<PaddingParams*>(branchParams[i]);
            layer                        =   new PaddingLayer(branchBuildParams.batch, branchBuildParams.height, branchBuildParams.width, branchBuildParams.channels,
                                                              paddingParams->top, paddingParams->down, paddingParams->left, paddingParams->right, paddingParams->paddingVal);
        }
        else
        {
            throw Exception(1, "layer type is not supported by [Res2BlockLayer]", __FILE__, __LINE__);
        }

        branchBuildParams.height       =   layer->outHeight;
        branchBuildParams.width        =   layer->outWidth;
        branchBuildParams.channels     =   layer->outChannel;
        branchBuildParams.inputNums    =   layer->outputNum;

        if(layer->workSpaceSize > this->workSpaceSize)
        {
            this->workSpaceSize = layer->workSpaceSize;
        }

        this->numWeights    =   this->numWeights + layer->numWeights;
        this->layerDetail   =   this->layerDetail.append(layer->layerDetail);

        branchLayers.push_back(layer);
    }

    if(branchBuildParams.height != params.height ||
            branchBuildParams.width != params.width ||
            branchBuildParams.channels != params.channels ||
            branchBuildParams.inputNums != params.inputNums)
    {
        throw Exception(1, "base output size is not equal with branch", __FILE__, __LINE__);
    }

    this->outHeight         =   params.height;
    this->outWidth          =   params.width;
    this->outChannel        =   params.channels;
    this->outputNum         =   params.inputNums;

    if(!BaseLayer::isPreviewMode)
    {
        this->output            =   new float[static_cast<size_t>(outputNum * this->batch)]();
    }

    this->layerDetail.append("========================================================================\n");
}

void Res2BlockLayer::loadAllWeigths(std::vector<float> &weights)
{

    if(weights.size() != this->numWeights)
    {
        throw Exception(1,"Res2Block weights load err. needed : " + std::to_string(this->numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__);
    }

    size_t ptr = 0;
    std::vector<float>::const_iterator first = weights.begin();

    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        if(baseLayers[i]->type == LayerType::CONVOLUTIONAL || baseLayers[i]->type == LayerType::CONNECTED || baseLayers[i]->type == LayerType::BATCHNORM)
        {
            size_t nums = baseLayers[i]->numWeights;

            std::vector<float> weights(first + static_cast<long long>(ptr), first + static_cast<long long>(ptr + nums));

            baseLayers[i]->loadAllWeigths(weights);

            ptr         =   ptr + nums;
        }
    }

    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        if(branchLayers[i]->type == LayerType::CONVOLUTIONAL || branchLayers[i]->type == LayerType::CONNECTED || branchLayers[i]->type == LayerType::BATCHNORM)
        {
            size_t nums = branchLayers[i]->numWeights;

            std::vector<float> weights(first + static_cast<long long>(ptr), first + static_cast<long long>(ptr + nums));

            branchLayers[i]->loadAllWeigths(weights);

            ptr         =   ptr + nums;
        }
    }
}

void Res2BlockLayer::forward(NetworkState &netState)
{

    /* TODO: batch */
    std::vector<float> inputX{netState.input, netState.input + netState.inputNum};

    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        baseLayers[i]->forward(netState);

        netState.input     =   baseLayers[i]->output;
        netState.inputNum  =   baseLayers[i]->outputNum;
    }

    netState.input         =    inputX.data();
    netState.inputNum      =    static_cast<int>(inputX.size());

    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        branchLayers[i]->forward(netState);

        netState.input     =   branchLayers[i]->output;
        netState.inputNum  =   branchLayers[i]->outputNum;

    }

    Blas::cpuAxpy(netState.inputNum, 1.f, baseLayers[baseLayers.size()-1]->output, 1, branchLayers[branchLayers.size()-1]->output, 1);
    Blas::cpuCopy(netState.inputNum, branchLayers[branchLayers.size()-1]->output, 1, this->output, 1);

    if(this->activation == ActivationType::NORM_CHAN)
    {
        Activations::activateArrayNormCh(this->output, this->outputNum, this->batch, this->outChannel,
                                         this->outWidth*this->outHeight, this->output);
    }
    else if(this->activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        Activations::activateArrayNormChSoftMax(this->output, this->outputNum, this->batch, this->outChannel,
                                                this->outWidth*this->outHeight, this->output,0);
    }
    else if(this->activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        Activations::activateArrayNormChSoftMax(this->output, this->outputNum, this->batch, this->outChannel,
                                                this->outWidth*this->outHeight, this->output,1);
    }
    else if(this->activation == ActivationType::NONE)
    {

    }
    else
    {
        if(actParams.size() > 0)
        {
            Activations::activateArray(this->output, this->outputNum, this->activation, this->supportAvx, actParams[0]);
        }
        else
        {
            Activations::activateArray(this->output, this->outputNum, this->activation, this->supportAvx);
        }
    }

    this->forwardTime = 0;

    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        this->forwardTime += baseLayers[i]->forwardTime;
    }

    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        this->forwardTime += branchLayers[i]->forwardTime;
    }

}

Res2BlockLayer::~Res2BlockLayer()
{
    for (size_t i = 0; i < baseLayers.size(); ++i)
    {
        if(baseLayers[i]!=nullptr)
        {
            if(baseLayers[i]->type == LayerType::CONVOLUTIONAL)
            {
                delete reinterpret_cast<ConvolutionalLayer*>(baseLayers[i]);
            }
            else if(baseLayers[i]->type == LayerType::MAXPOOL)
            {
                delete reinterpret_cast<MaxPoolLayer*>(baseLayers[i]);
            }
            else if(baseLayers[i]->type == LayerType::CONNECTED)
            {
                delete reinterpret_cast<ConnectedLayer*>(baseLayers[i]);
            }
            else if(baseLayers[i]->type == LayerType::BATCHNORM)
            {
                delete reinterpret_cast<BatchNormLayer*>(baseLayers[i]);
            }
            else if(baseLayers[i]->type == LayerType::LOCAL_AVGPOOL)
            {
                delete reinterpret_cast<LocalAvgPoolLayer*>(baseLayers[i]);
            }

            baseLayers[i] = nullptr;
        }

        if(i == (baseLayers.size()-1))
        {
            baseLayers.clear();
        }
    }

    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        if(branchLayers[i]!=nullptr)
        {
            if(branchLayers[i]->type == LayerType::CONVOLUTIONAL)
            {
                delete reinterpret_cast<ConvolutionalLayer*>(branchLayers[i]);
            }
            else if(branchLayers[i]->type == LayerType::MAXPOOL)
            {
                delete reinterpret_cast<MaxPoolLayer*>(branchLayers[i]);
            }
            else if(branchLayers[i]->type == LayerType::CONNECTED)
            {
                delete reinterpret_cast<ConnectedLayer*>(branchLayers[i]);
            }
            else if(branchLayers[i]->type == LayerType::BATCHNORM)
            {
                delete reinterpret_cast<BatchNormLayer*>(branchLayers[i]);
            }
            else if(branchLayers[i]->type == LayerType::LOCAL_AVGPOOL)
            {
                delete reinterpret_cast<LocalAvgPoolLayer*>(branchLayers[i]);
            }
            else if(branchLayers[i]->type == LayerType::PADDING)
            {
                delete reinterpret_cast<PaddingLayer*>(branchLayers[i]);
            }

            branchLayers[i] = nullptr;
        }

        if(i == (branchLayers.size()-1))
        {
            branchLayers.clear();
        }
    }
}
}
