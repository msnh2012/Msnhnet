#include "Msnhnet/layers/MsnhAddBlockLayer.h"
namespace Msnhnet
{
AddBlockLayer::AddBlockLayer(const int &batch, NetBuildParams &params, std::vector<std::vector<BaseParams *> > &branchParams, ActivationType &activation, const std::vector<float> &actParams)
{
    this->_type          =   LayerType::ADD_BLOCK;
    this->_layerName     =   "AddBlock        ";
    this->_activation    =   activation;
    this->_actParams     =   actParams;

    this->_batch         =   batch;
    this->_width         =   params.width;
    this->_height        =   params.height;
    this->_channel       =   params.channels;

    BaseLayer *layer    =   nullptr;

    NetBuildParams  branchBuildParams = params;

    this->_layerDetail.append("=============================== AddBlock ===============================\n");

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
                    throw Exception(1, "Layer before convolutional layer must output image", __FILE__, __LINE__, __FUNCTION__);
                }

                ConvParams* convParams      =   reinterpret_cast<ConvParams*>(branchParams[i][j]);
                layer                       =   new ConvolutionalLayer(branchBuildParams.batch, 1, branchBuildParams.height, branchBuildParams.width, branchBuildParams.channels,
                                                                       convParams->filters,convParams->groups,convParams->kSizeX, convParams->kSizeY,convParams->strideX, convParams->strideY,
                                                                       convParams->dilationX,convParams->dilationY,convParams->paddingX, convParams->paddingY, convParams->activation, convParams->actParams, convParams->batchNorm, convParams->useBias,
                                                                       0,0,0,0,convParams->antialiasing, nullptr, 0,0);
                if(i == 0 && j == 0)
                {
                    this->_inputNum = layer->getInputNum();
                }

            }
            else if(branchParams[i][j]->type == LayerType::PADDING)
            {
                PaddingParams *paddingParams =   reinterpret_cast<PaddingParams*>(branchParams[i][j]);
                layer                        =   new PaddingLayer(branchBuildParams.batch, branchBuildParams.height, branchBuildParams.width, branchBuildParams.channels,
                                                                  paddingParams->top, paddingParams->down, paddingParams->left, paddingParams->right, paddingParams->paddingVal);
                if(i == 0 && j == 0)
                {
                    this->_inputNum = layer->getInputNum();
                }
            }
            else if(branchParams[i][j]->type == LayerType::CONNECTED)
            {
                ConnectParams *connectParams=   reinterpret_cast<ConnectParams*>(branchParams[i][j]);
                layer                       =   new ConnectedLayer(branchBuildParams.batch, 1, branchBuildParams.inputNums, connectParams->output, connectParams->activation, connectParams->actParams,
                                                                   connectParams->batchNorm, connectParams->useBias);
                if(i == 0 && j == 0)
                {
                    this->_inputNum = layer->getInputNum();
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
                    this->_inputNum = layer->getInputNum();
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
                    this->_inputNum = layer->getInputNum();
                }
            }
            else if(branchParams[i][j]->type == LayerType::BATCHNORM)
            {
                BatchNormParams *batchNormParams=   reinterpret_cast<BatchNormParams*>(branchParams[i][j]);
                layer                           =   new BatchNormLayer(branchBuildParams.batch, branchBuildParams.width, branchBuildParams.height, branchBuildParams.channels, batchNormParams->activation, batchNormParams->actParams);
                if(i == 0 && j == 0)
                {
                    this->_inputNum = layer->getInputNum();
                }
            }
            else if(branchParams[i][j]->type == LayerType::EMPTY)
            {
                layer                           =   new EmptyLayer(branchBuildParams.batch, branchBuildParams.width, branchBuildParams.height, branchBuildParams.channels);
                if(i == 0 && j == 0)
                {
                    this->_inputNum = layer->getInputNum();
                }
            }
            else
            {
                throw Exception(1, "layer type is not supported by [AddBlockLayer]", __FILE__, __LINE__, __FUNCTION__);
            }

            branchBuildParams.height       =   layer->getOutHeight();
            branchBuildParams.width        =   layer->getOutWidth();
            branchBuildParams.channels     =   layer->getOutChannel();
            branchBuildParams.inputNums    =   layer->getOutputNum();

            if(layer->getWorkSpaceSize() > this->getWorkSpaceSize())
            {
                this->_workSpaceSize = layer->getWorkSpaceSize();
            }

            if(layer->getMaxOutputNum() >this->_maxOutputNum)
            {
                this->_maxOutputNum  = layer->getMaxOutputNum();
            }

            this->_numWeights    =   this->_numWeights + layer->getNumWeights();
            this->_layerDetail   =   this->_layerDetail.append(layer->getLayerDetail());

            tmpLayers.push_back(layer);

            if(layer->getMemReUse()==0)
            {
                layer->mallocMemory();
            }

            if(j == (branchParams[i].size()-1))
            {
                layer->mallocMemory();
            }

            layer->setIsBranchLayer(true);

            if(j == 0)
            {
                layer->setBranchFirst(true);
            }

            if(j==(branchParams[i].size()-1))
            {
                layer->setBranchLast(true);
            }
        }

        this->_layerDetail.append("\n");

        branchLayers.push_back(tmpLayers);

    }

    for (size_t i = 1; i < branchLayers.size(); ++i)
    {
        if(branchLayers[i][branchLayers[i].size()-1]->getHeight()     != branchLayers[i-1][branchLayers[i-1].size()-1]->getHeight()||
                branchLayers[i][branchLayers[i].size()-1]->getWidth()      != branchLayers[i-1][branchLayers[i-1].size()-1]->getWidth()||
                branchLayers[i][branchLayers[i].size()-1]->getOutChannel() != branchLayers[i-1][branchLayers[i-1].size()-1]->getOutChannel()||
                branchLayers[i][branchLayers[i].size()-1]->getOutputNum()  != branchLayers[i-1][branchLayers[i-1].size()-1]->getOutputNum())
        {
            throw Exception(1, "branch's outputs size is not equal", __FILE__, __LINE__, __FUNCTION__);
        }
    }

    this->_outHeight         =   branchBuildParams.height;
    this->_outWidth          =   branchBuildParams.width;
    this->_outChannel        =   branchBuildParams.channels;
    this->_outputNum         =   branchBuildParams.inputNums;

    this->_layerDetail.append("========================================================================\n");
}

void AddBlockLayer::loadAllWeigths(std::vector<float> &weights)
{
    if(weights.size() != this->_numWeights)
    {
        throw Exception(1,"AddBlockLayer weights load err. needed : " + std::to_string(this->_numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__, __FUNCTION__);
    }

    size_t ptr = 0;
    std::vector<float>::const_iterator first = weights.begin();

    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        for (size_t j = 0; j < branchLayers[i].size(); ++j)
        {
            if(branchLayers[i][j]->type() == LayerType::CONVOLUTIONAL || branchLayers[i][j]->type() == LayerType::CONNECTED || branchLayers[i][j]->type() == LayerType::BATCHNORM ||
                    branchLayers[i][j]->type() == LayerType::ADD_BLOCK || branchLayers[i][j]->type() == LayerType::CONCAT_BLOCK)
            {
                size_t nums = branchLayers[i][j]->getNumWeights();

                std::vector<float> weights(first + static_cast<long long>(ptr), first + static_cast<long long>(ptr + nums));

                branchLayers[i][j]->loadAllWeigths(weights);

                ptr         =   ptr + nums;
            }
        }
    }
}

void AddBlockLayer::mallocMemory()
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

void AddBlockLayer::forward(NetworkState &netState)
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

    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        netState.input         =    inputX.data();
        netState.inputNum      =    static_cast<int>(inputX.size());

        for (size_t j = 0; j < branchLayers[i].size(); ++j)
        {
            branchLayers[i][j]->forward(netState);

            if(branchLayers[i][j]->getMemReUse()==0)

            {
                netState.input     =   branchLayers[i][j]->getOutput();
            }

            netState.inputNum  =   branchLayers[i][j]->getOutputNum();
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

    for (size_t i = 1; i < branchLayers.size(); ++i)
    {

        Blas::cpuAxpy(netState.inputNum, 1.f, branchLayers[i-1][branchLayers[i-1].size()-1]->getOutput(),
                1, branchLayers[i][branchLayers[i].size()-1]->getOutput(), 1);

    }
    Blas::cpuCopy(netState.inputNum, branchLayers[branchLayers.size()-1][branchLayers[branchLayers.size()-1].size()-1]->getOutput(), 1, layerOutput, 1);

    if(this->_activation == ActivationType::NORM_CHAN)
    {
        Activations::activateArrayNormCh(layerOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                         this->_outWidth*this->_outHeight, layerOutput);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        Activations::activateArrayNormChSoftMax(layerOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, layerOutput,0);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        Activations::activateArrayNormChSoftMax(layerOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, layerOutput,1);
    }
    else if(this->_activation == ActivationType::NONE)
    {

    }
    else
    {
        if(_actParams.size() > 0)
        {
            Activations::activateArray(layerOutput, this->_outputNum*this->_batch, this->_activation, this->supportAvx, _actParams[0]);
        }
        else
        {
            Activations::activateArray(layerOutput, this->_outputNum*this->_batch, this->_activation, this->supportAvx);
        }
    }

    this->_forwardTime = 0;

    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        for (size_t j = 0; j < branchLayers[i].size(); ++j)
        {
            this->_forwardTime += branchLayers[i][j]->getForwardTime();
        }
    }
}

#ifdef USE_GPU
void AddBlockLayer::forwardGPU(NetworkState &netState)
{
    /* TODO: batch */

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
    int     inputXNum   = netState.inputNum;

    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        netState.input         =    inputX;
        netState.inputNum      =    inputXNum;

        for (size_t j = 0; j < branchLayers[i].size(); ++j)
        {
            branchLayers[i][j]->forwardGPU(netState);

            if(branchLayers[i][j]->getMemReUse()==0)

            {
                netState.input     =   branchLayers[i][j]->getGpuOutput();
            }
            netState.inputNum  =   branchLayers[i][j]->getOutputNum();
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

    for (size_t i = 1; i < branchLayers.size(); ++i)
    {

        BlasGPU::gpuAxpy(netState.inputNum, 1.f, branchLayers[i-1][branchLayers[i-1].size()-1]->getGpuOutput(),
                1, branchLayers[i][branchLayers[i].size()-1]->getGpuOutput(), 1);

    }
    BlasGPU::gpuCopy(netState.inputNum, branchLayers[branchLayers.size()-1][branchLayers[branchLayers.size()-1].size()-1]->getGpuOutput(), 1, layerGpuOutput, 1);

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

    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        for (size_t j = 0; j < branchLayers[i].size(); ++j)
        {
            this->_forwardTime += branchLayers[i][j]->getForwardTime();
        }
    }

    Cuda::freeCuda(inputX);
}
#endif

AddBlockLayer::~AddBlockLayer()
{
    for (size_t i = 0; i < branchLayers.size(); ++i)
    {
        for (size_t j = 0; j < branchLayers[i].size(); ++j)
        {
            if(branchLayers[i][j]!=nullptr)
            {
                if(branchLayers[i][j]->type() == LayerType::CONVOLUTIONAL)
                {
                    delete reinterpret_cast<ConvolutionalLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type() == LayerType::MAXPOOL)
                {
                    delete reinterpret_cast<MaxPoolLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type() == LayerType::CONNECTED)
                {
                    delete reinterpret_cast<ConnectedLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type() == LayerType::BATCHNORM)
                {
                    delete reinterpret_cast<BatchNormLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type() == LayerType::LOCAL_AVGPOOL)
                {
                    delete reinterpret_cast<LocalAvgPoolLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type() == LayerType::EMPTY)
                {
                    delete reinterpret_cast<EmptyLayer*>(branchLayers[i][j]);
                }
                else if(branchLayers[i][j]->type() == LayerType::PADDING)
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
