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

    this->_layerDetail   = "Activate Layer: " + Activations::getActivationStr(this->activation()) + "\n";
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
                this->_output           =   new float[static_cast<size_t>(this->_batch*this->_outputNum)]();
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

        ActivationsGPU::gpuActivateArray(layerGpuInput,
                                         _outputNum*_batch,
                                         _activation
                                         );

    }
    else    

    {
        ActivationsGPU::gpuActivateArray(this->_gpuOutput, 

                                         _outputNum*_batch,
                                         _activation
                                         );
    }

    this->recordCudaStop();
}
#endif
}
