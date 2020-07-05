#include "Msnhnet/layers/MsnhAddBlockLayer.h"
namespace Msnhnet
{
AddBlockLayer::AddBlockLayer(const int &batch, NetBuildParams &params, std::vector<std::vector<BaseParams *> > &branchParams, ActivationType &activation, const std::vector<float> &actParams)
{
    this->type          =   LayerType::ADD_BLOCK;
    this->layerName     =   "AddBlock        ";
    this->activation    =   activation;
    this->actParams     =   actParams;

   this->batch         =   batch;
    this->width         =   params.width;
    this->height        =   params.height;
    this->channel       =   params.channels;

   BaseLayer *layer    =   nullptr;

   NetBuildParams  branchBuildParams = params;

   this->layerDetail.append("=============================== AddBlock ===============================\n");

   for (size_t i = 0; i < branchParams.size(); ++i)
    {
        branchBuildParams = params;  

       std::vector<BaseLayer* > tmpLayers;
        for (size_t j = 0; j < branchParams[i].size(); ++j)
        {

           if(branchParams[i][j]->type == LayerType::CONVOLUTIONAL)
            {
                if(branchBuildParams.height ==0 || branchBuildParams.width == 0 || branchBuildParams.channels == 0)
                {
                    throw Exception(1, "Layer before convolutional layer must output image", __FILE__, __LINE__);
                }

               ConvParams* convParams      =   reinterpret_cast<ConvParams*>(branchParams[i][j]);
                layer                       =   new ConvolutionalLayer(branchBuildParams.batch, 1, branchBuildParams.height, branchBuildParams.width, branchBuildParams.channels,
                                                                       convParams->filters,convParams->groups,convParams->kSizeX, convParams->kSizeY,convParams->strideX, convParams->strideY,
                                                                       convParams->dilationX,convParams->dilationY,convParams->paddingX, convParams->paddingY, convParams->activation, convParams->actParams, convParams->batchNorm, convParams->useBias,
                                                                       0,0,0,0,convParams->antialiasing, nullptr, 0,0);
                if(i == 0 && j == 0)
                {
                    this->inputNum = layer->inputNum;
                }

           }
            else if(branchParams[i][j]->type == LayerType::PADDING)
            {
                PaddingParams *paddingParams =   reinterpret_cast<PaddingParams*>(branchParams[i][j]);
                layer                        =   new PaddingLayer(branchBuildParams.batch, branchBuildParams.height, branchBuildParams.width, branchBuildParams.channels,
                                                                  paddingParams->top, paddingParams->down, paddingParams->left, paddingParams->right, paddingParams->paddingVal);
                if(i == 0 && j == 0)
                {
                    this->inputNum = layer->inputNum;
                }
            }
            else if(branchParams[i][j]->type == LayerType::CONNECTED)
            {
                ConnectParams *connectParams=   reinterpret_cast<ConnectParams*>(branchParams[i][j]);
                layer                       =   new ConnectedLayer(branchBuildParams.batch, 1, branchBuildParams.inputNums, connectParams->output, connectParams->activation, connectParams->actParams,
                                                                   connectParams->batchNorm);
                if(i == 0 && j == 0)
                {
                    this->inputNum = layer->inputNum;
                }
            }
            else if(branchParams[i][j]->type == LayerType::MAXPOOL)
            {
                MaxPoolParams *maxPoolParams=   reinterpret_cast<MaxPoolParams*>(branchParams[i][j]);
                layer                       =   new MaxPoolLayer(branchBuildParams.batch, branchBuildParams.height, branchBuildParams.width, branchBuildParams.channels, maxPoolParams->kSizeX, maxPoolParams->kSizeY,
                                                                 maxPoolParams->strideX, maxPoolParams->strideY, maxPoolParams->paddingX, maxPoolParams->paddingY,
                                                                 maxPoolParams->maxPoolDepth, maxPoolParams->outChannels, maxPoolParams->ceilMode, 0);
                if(i == 0 && j == 0)
                {
                    this->inputNum = layer->inputNum;
                }
            }
            else if(branchParams[i][j]->type == LayerType::LOCAL_AVGPOOL)
            {
                LocalAvgPoolParams *localAvgPoolParams  =   reinterpret_cast<LocalAvgPoolParams*>(branchParams[i][j]);
                layer                                   =   new LocalAvgPoolLayer(branchBuildParams.batch, branchBuildParams.height, branchBuildParams.width, branchBuildParams.channels,
                                                                                  localAvgPoolParams->kSizeX, localAvgPoolParams->kSizeY, localAvgPoolParams->strideX, localAvgPoolParams->strideY,
                                                                                  localAvgPoolParams->paddingX, localAvgPoolParams->paddingY, localAvgPoolParams->ceilMode, 0);
                if(i == 0 && j == 0)
                {
                    this->inputNum = layer->inputNum;
                }
            }
            else if(branchParams[i][j]->type == LayerType::BATCHNORM)
            {
                BatchNormParams *batchNormParams=   reinterpret_cast<BatchNormParams*>(branchParams[i][j]);
                layer                           =   new BatchNormLayer(branchBuildParams.batch, branchBuildParams.width, branchBuildParams.height, branchBuildParams.channels, batchNormParams->activation, batchNormParams->actParams);
                if(i == 0 && j == 0)
                {
                    this->inputNum = layer->inputNum;
                }
            }
            else if(branchParams[i][j]->type == LayerType::EMPTY)
            {
                layer                           =   new EmptyLayer(branchBuildParams.batch, branchBuildParams.width, branchBuildParams.height, branchBuildParams.channels);
                if(i == 0 && j == 0)
                {
                    this->inputNum = layer->inputNum;
                }
            }
            else if(branchParams[i][j]->type == LayerType::ADD_BLOCK)
            {
                AddBlockParams *addBlockParams          =   reinterpret_cast<AddBlockParams*>(branchParams[i][j]);
                layer                                   =   new AddBlockLayer(1, params, addBlockParams->branchParams, addBlockParams->activation, addBlockParams->actParams);
                if(i == 0 && j == 0)
                {
                    this->inputNum = layer->inputNum;
                }
            }
            else if(branchParams[i][j]->type == LayerType::CONCAT_BLOCK)
            {
                ConcatBlockParams *concatBlockParams    =   reinterpret_cast<ConcatBlockParams*>(branchParams[i][j]);
                layer                                   =   new ConcatBlockLayer(1, params, concatBlockParams->branchParams, concatBlockParams->activation, concatBlockParams->actParams);
                if(i == 0 && j == 0)
                {
                    this->inputNum = layer->inputNum;
                }
            }
            else
            {
                throw Exception(1, "layer type is not supported by [AddBlockLayer]", __FILE__, __LINE__);
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

           tmpLayers.push_back(layer);
        }
        this->layerDetail.append("\n");

       branchLayers.push_back(tmpLayers);

   }

   for (size_t i = 1; i < branchLayers.size(); ++i)
    {
        if(branchLayers[i][branchLayers[i].size()-1]->height     != branchLayers[i-1][branchLayers[i-1].size()-1]->height||
                branchLayers[i][branchLayers[i].size()-1]->width      != branchLayers[i-1][branchLayers[i-1].size()-1]->width||
                branchLayers[i][branchLayers[i].size()-1]->outChannel != branchLayers[i-1][branchLayers[i-1].size()-1]->outChannel||
                branchLayers[i][branchLayers[i].size()-1]->outputNum  != branchLayers[i-1][branchLayers[i-1].size()-1]->outputNum)
        {
            throw Exception(1, "branch's outputs size is not equal", __FILE__, __LINE__);
        }
    }

   this->outHeight         =   branchBuildParams.height;
    this->outWidth          =   branchBuildParams.width;
    this->outChannel        =   branchBuildParams.channels;
    this->outputNum         =   branchBuildParams.inputNums;

   if(!BaseLayer::isPreviewMode)
    {
        this->output            =   new float[static_cast<size_t>(outputNum * this->batch)]();
    }

   this->layerDetail.append("========================================================================\n");
}

void AddBlockLayer::loadAllWeigths(std::vector<float> &weights)
{
    if(weights.size() != this->numWeights)
    {
        throw Exception(1,"AddBlockLayer weights load err. needed : " + std::to_string(this->numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__);
    }

   size_t ptr = 0;
    std::vector<float>::const_iterator first = weights.begin();

   for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        for (size_t j = 0; j < branchLayers[i].size(); ++j)
        {
            if(branchLayers[i][j]->type == LayerType::CONVOLUTIONAL || branchLayers[i][j]->type == LayerType::CONNECTED || branchLayers[i][j]->type == LayerType::BATCHNORM ||
                    branchLayers[i][j]->type == LayerType::ADD_BLOCK || branchLayers[i][j]->type == LayerType::CONCAT_BLOCK)
            {
                size_t nums = branchLayers[i][j]->numWeights;

               std::vector<float> weights(first + static_cast<long long>(ptr), first + static_cast<long long>(ptr + nums));

               branchLayers[i][j]->loadAllWeigths(weights);

               ptr         =   ptr + nums;
            }
        }
    }
}

void AddBlockLayer::forward(NetworkState &netState)
{

   /* TODO: batch */
    std::vector<float> inputX{netState.input, netState.input + netState.inputNum};

   for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        netState.input         =    inputX.data();
        netState.inputNum      =    static_cast<int>(inputX.size());

       for (size_t j = 0; j < branchLayers[i].size(); ++j)
        {
            branchLayers[i][j]->forward(netState);

           netState.input     =   branchLayers[i][j]->output;
            netState.inputNum  =   branchLayers[i][j]->outputNum;
        }

   }

   for (size_t i = 1; i < branchLayers.size(); ++i)
    {

       Blas::cpuAxpy(netState.inputNum, 1.f, branchLayers[i-1][branchLayers[i-1].size()-1]->output,
                1, branchLayers[i][branchLayers[i].size()-1]->output, 1);

   }
    Blas::cpuCopy(netState.inputNum, branchLayers[branchLayers.size()-1][branchLayers[branchLayers.size()-1].size()-1]->output, 1, this->output, 1);

   if(this->activation == ActivationType::NORM_CHAN)
    {
        Activations::activateArrayNormCh(this->output, this->outputNum*this->batch, this->batch, this->outChannel,
                                         this->outWidth*this->outHeight, this->output);
    }
    else if(this->activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        Activations::activateArrayNormChSoftMax(this->output, this->outputNum*this->batch, this->batch, this->outChannel,
                                                this->outWidth*this->outHeight, this->output,0);
    }
    else if(this->activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        Activations::activateArrayNormChSoftMax(this->output, this->outputNum*this->batch, this->batch, this->outChannel,
                                                this->outWidth*this->outHeight, this->output,1);
    }
    else if(this->activation == ActivationType::NONE)
    {

   }
    else
    {
        if(actParams.size() > 0)
        {
            Activations::activateArray(this->output, this->outputNum*this->batch, this->activation, actParams[0]);
        }
        else
        {
            Activations::activateArray(this->output, this->outputNum*this->batch, this->activation);
        }
    }

   this->forwardTime = 0;

   for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        for (size_t j = 0; j < branchLayers[i].size(); ++j)
        {
            this->forwardTime += branchLayers[i][j]->forwardTime;
        }
    }
}

AddBlockLayer::~AddBlockLayer()
{
    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        for (size_t j = 0; j < branchLayers[i].size(); ++j)
        {
            if(branchLayers[i][j]!=nullptr)
            {
                if(branchLayers[i][j]->type == LayerType::CONVOLUTIONAL)
                {
                    delete reinterpret_cast<ConvolutionalLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type == LayerType::MAXPOOL)
                {
                    delete reinterpret_cast<MaxPoolLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type == LayerType::CONNECTED)
                {
                    delete reinterpret_cast<ConnectedLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type == LayerType::BATCHNORM)
                {
                    delete reinterpret_cast<BatchNormLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type == LayerType::LOCAL_AVGPOOL)
                {
                    delete reinterpret_cast<LocalAvgPoolLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type == LayerType::EMPTY)
                {
                    delete reinterpret_cast<EmptyLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type == LayerType::PADDING)
                {
                    delete reinterpret_cast<PaddingLayer*>(branchLayers[i][j]);
                }

               branchLayers[i][j] = nullptr;
            }
        }

       if(i == (branchLayers.size()-1))
        {
            branchLayers.clear();
        }
    }
}
}
