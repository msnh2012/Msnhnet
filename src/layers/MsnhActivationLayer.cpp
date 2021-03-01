#include "Msnhnet/layers/MsnhActivationLayer.h"

namespace Msnhnet
{
ActivationLayer::ActivationLayer(const int &batch, const int &width, const int &height, const int &channel, const int &inputNum, const ActivationType &activation, const std::vector<float> &actParams)
{
    this->_layerName     = "Activate        ";
    this->_type          = LayerType::ACTIVE;
    this->_inputNum      = inputNum;
    this->_outputNum     = inputNum;
    this->_batch         = batch;
    this->_activation    = activation;

    this->_height        = height;
    this->_width         = width;
    this->_channel       = channel;
    this->_actParams     = actParams;

    this->_outHeight     = this->_height;
    this->_outWidth      = this->_width;
    this->_outChannel    = this->_channel;
    this->_maxOutputNum  = this->_batch*this->_outputNum;

    if(activation==ActivationType::PRELU) 

    {
        this->_nPreluWeights = channel;
    }
    else
    {
        this->_nPreluWeights = 0;
    }

    this->_numWeights = this->_nPreluWeights;

    this->_layerDetail   = "Activate Layer: " + Activations::getActivationStr(this->activation()) + "\n";
}

ActivationLayer::~ActivationLayer()
{
    if(this->_activation==ActivationType::PRELU) 

    {
        releaseArr(this->_preluWeights);

#ifdef USE_GPU
        Cuda::freeCuda(this->_gpuPreluWeights);
#endif
    }
}

void ActivationLayer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();

    float* layerInput = netState.getInput();

    if(this->_layerIndex == 0) 

    {
        if(this->_memReUse == 1) 

        {
            Blas::cpuCopy(this->_batch*this->_inputNum, netState.input, 1, layerInput, 1);
        }
        else 

        {
            Blas::cpuCopy(this->_batch*this->_inputNum, netState.input, 1, this->_output, 1);
        }

    }
    else 

    {
        if(netState.net->layers[this->_layerIndex - 1]->getMemReUse() == 0)

        {
            if(this->_memReUse == 1) 

            {
                Blas::cpuCopy(this->_batch*this->_inputNum, netState.input, 1, layerInput, 1);
            }
            else 

            {
                Blas::cpuCopy(this->_batch*this->_inputNum, netState.input, 1, this->_output, 1);
            }
        }
        else 

        {
            if(this->_memReUse == 1) 

            {

            }
            else 

            {
                Blas::cpuCopy(this->_batch*this->_inputNum, layerInput, 1, this->_output, 1);
            }
        }
    }

    if(this->_activation == ActivationType::NONE)
    {
        this->_forwardTime = TimeUtil::getElapsedTime(st);
        return;
    }

    if(this->_memReUse==1) 

    {
        if(this->_activation == ActivationType::NORM_CHAN)
        {
            Activations::activateArrayNormCh(layerInput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                             this->_outWidth*this->_outHeight, layerInput);
        }
        else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
        {
            Activations::activateArrayNormChSoftMax(layerInput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                    this->_outWidth*this->_outHeight, layerInput,0);
        }
        else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
        {
            Activations::activateArrayNormChSoftMax(layerInput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                    this->_outWidth*this->_outHeight, layerInput,1);
        }
        else if(this->_activation == ActivationType::PRELU) 

        {
            Activations::activatePRelu(layerInput,this->_batch, this->_outChannel, this->_preluWeights, this->_outWidth*this->_outHeight,this->supportAvx);
        }
        else if(this->_activation == ActivationType::NONE)
        {

        }
        else
        {

            if(this->_actParams.size()>0)
            {
                Activations::activateArray(layerInput,
                                           _outputNum*_batch,
                                           _activation,
                                           this->supportAvx,
                                           this->_actParams[0]
                        );
            }
            else
            {
                Activations::activateArray(layerInput,
                                           _outputNum*_batch,
                                           _activation,
                                           this->supportAvx
                                           );
            }
        }

    }
    else    

    {
        if(this->_activation == ActivationType::NORM_CHAN)
        {
            Activations::activateArrayNormCh(this->_output, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                             this->_outWidth*this->_outHeight, this->_output);
        }
        else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
        {
            Activations::activateArrayNormChSoftMax(this->_output, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                    this->_outWidth*this->_outHeight, this->_output,0);
        }
        else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
        {
            Activations::activateArrayNormChSoftMax(this->_output, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                    this->_outWidth*this->_outHeight, this->_output,1);
        }
        else if(this->_activation == ActivationType::PRELU) 

        {
            Activations::activatePRelu(this->_output,this->_batch, this->_outChannel, this->_preluWeights, this->_outWidth*this->_outHeight, this->supportAvx);
        }
        else if(this->_activation == ActivationType::NONE)
        {

        }
        else
        {
            if(this->_actParams.size()>0)
            {
                Activations::activateArray(this->_output, 

                                           _outputNum*_batch,
                                           _activation,
                                           this->supportAvx,
                                           this->_actParams[0]
                        );
            }
            else
            {
                Activations::activateArray(this->_output, 

                                           _outputNum*_batch,
                                           _activation,
                                           this->supportAvx
                                           );
            }
        }

    }

    this->_forwardTime = TimeUtil::getElapsedTime(st);
}

void ActivationLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu)

            {

                this->_output   = MemoryManager::effcientNew<float>(static_cast<size_t>(this->_batch*this->_outputNum));
            }
#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuOutput        =   Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced   =   true;
        }
    }
    this->_memReUse         =  0;
}

void ActivationLayer::loadAllWeigths(std::vector<float> &weights)
{
    if(this->_activation == ActivationType::PRELU) 

    {
        if(weights.size() != this->_numWeights)
        {
            throw Exception(1,"PRelu weights load err. needed : " + std::to_string(this->_numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__, __FUNCTION__);
        }

        if(!BaseLayer::isPreviewMode)
        {

            this->_preluWeights   = MemoryManager::effcientNew<float>(static_cast<size_t>(this->_nPreluWeights));

            loadPreluWeights(weights.data(), this->_nPreluWeights);
#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuPreluWeights = Cuda::makeCudaArray(this->_preluWeights, this->_nPreluWeights);
            }

            if(BaseLayer::onlyUseGpu) 

            {
                releaseArr(this->_preluWeights);
            }
#endif
        }
    }

    this->_weightsLoaded = true;
}

void ActivationLayer::saveAllWeights(const int &mainIdx, const int &branchIdx, const int &branchIdx1)
{
    if(this->_activation == ActivationType::PRELU) 

    {
        if(BaseLayer::isPreviewMode)
        {
            throw Exception(1,"Activation preview mode can't save weights.", __FILE__, __LINE__, __FUNCTION__);
        }

        if(!this->_weightsLoaded)
        {
            throw Exception(1,"Activation weights had not been loaded yet.", __FILE__, __LINE__, __FUNCTION__);
        }

        std::string name = "";

        if(branchIdx!=-1)
        {
            name = "_" + std::to_string(mainIdx) + "_" + std::to_string(branchIdx) + "_" + std::to_string(branchIdx1) +".txt";
        }
        else
        {
            name = std::to_string(this->_layerIndex)+".txt";
        }

#ifdef USE_GPU
        if(BaseLayer::onlyUseGpu) 

        {
            Cuda::pullCudaArray(this->_gpuPreluWeights, this->_preluWeights, this->_nPreluWeights);
        }
#endif

        if(this->_preluWeights==nullptr)
        {
            throw Exception(1,"preluWeights weights err.", __FILE__, __LINE__, __FUNCTION__);
        }

        std::vector<float> preluWeightVec(this->_preluWeights, this->_preluWeights + this->_nPreluWeights);

        std::string preluWeightsName = "preluWeights"+name;
        Msnhnet::IO::saveVector<float>(preluWeightVec,preluWeightsName.c_str(),"\n");
    }
}

void ActivationLayer::loadPreluWeights(float * const &weights, const int &len)
{
    if(len != this->_nPreluWeights)
    {
        throw Exception(1, "load preluWeights data len error",__FILE__,__LINE__, __FUNCTION__);
    }
    Blas::cpuCopy(len, weights, 1, this->_preluWeights,1);
}

#ifdef USE_GPU
void ActivationLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

    float* layerGpuInput = netState.getGpuInput();

    if(this->_layerIndex == 0) 

    {
        if(this->_memReUse == 1) 

        {
            BlasGPU::gpuCopy(this->_batch*this->_inputNum, netState.input, 1, layerGpuInput, 1);
        }
        else 

        {
            BlasGPU::gpuCopy(this->_batch*this->_inputNum, netState.input, 1, this->_gpuOutput, 1);
        }

    }
    else 

    {
        if(netState.net->layers[this->_layerIndex - 1]->getMemReUse() == 0)

        {
            if(this->_memReUse == 1) 

            {
                BlasGPU::gpuCopy(this->_batch*this->_inputNum, netState.input, 1, layerGpuInput, 1);
            }
            else 

            {
                BlasGPU::gpuCopy(this->_batch*this->_inputNum, netState.input, 1, this->_gpuOutput, 1);
            }
        }
        else 

        {
            if(this->_memReUse == 1) 

            {

            }
            else 

            {
                BlasGPU::gpuCopy(this->_batch*this->_inputNum, layerGpuInput, 1, this->_gpuOutput, 1);
            }
        }
    }

    if(this->_memReUse==1) 

    {

        if(this->_activation == ActivationType::NORM_CHAN)
        {
            ActivationsGPU::gpuActivateArrayNormCh(layerGpuInput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                   this->_outWidth*this->_outHeight, layerGpuInput);
        }
        else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
        {
            ActivationsGPU::gpuActivateArrayNormChSoftMax(layerGpuInput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                          this->_outWidth*this->_outHeight, layerGpuInput,0);
        }
        else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
        {
            ActivationsGPU::gpuActivateArrayNormChSoftMax(layerGpuInput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                          this->_outWidth*this->_outHeight, layerGpuInput,1);
        }
        else if(this->_activation == ActivationType::PRELU)
        {
            ActivationsGPU::gpuActivatePRelu(layerGpuInput,this->_batch, this->_outChannel, this->_gpuPreluWeights, this->_outWidth*this->_outHeight);
        }
        else if(this->_activation == ActivationType::NONE)
        {

        }
        else
        {
            if(this->_actParams.size()>0)
            {
                ActivationsGPU::gpuActivateArray(layerGpuInput,
                                                 _outputNum*_batch,
                                                 _activation,
                                                 this->_actParams[0]
                        );
            }
            else
            {
                ActivationsGPU::gpuActivateArray(layerGpuInput,
                                                 _outputNum*_batch,
                                                 _activation
                                                 );
            }
        }

    }
    else    

    {

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
        else if(this->_activation == ActivationType::PRELU)
        {
            ActivationsGPU::gpuActivatePRelu(this->_gpuOutput,this->_batch, this->_outChannel, this->_gpuPreluWeights, this->_outWidth*this->_outHeight);
        }
        else if(this->_activation == ActivationType::NONE)
        {

        }
        else
        {
            if(this->_actParams.size()>0)
            {
                ActivationsGPU::gpuActivateArray(this->_gpuOutput,
                                                 _outputNum*_batch,
                                                 _activation,
                                                 this->_actParams[0]
                        );
            }
            else
            {
                ActivationsGPU::gpuActivateArray(this->_gpuOutput,
                                                 _outputNum*_batch,
                                                 _activation
                                                 );
            }
        }

    }

    this->recordCudaStop();
}
#endif
}