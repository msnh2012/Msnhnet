#include "Msnhnet/layers/MsnhReductionLayer.h"
namespace Msnhnet
{
ReductionLayer::ReductionLayer(const int &batch, const int &height, const int &width, const int &channel, const int &axis, const ReductionType &reductionType)
{
    this->_layerName =  "Reduction       ";
    this->_type      =   LayerType::REDUCTION;

    this->_batch    =   batch;
    this->_channel  =   channel;
    this->_height   =   height;
    this->_width    =   width;

    this->_axis     =   axis;

    this->_reductionType     =   reductionType;

    if(this->_axis == -1)
    {
        this->_outWidth     = 1;
        this->_outHeight    = 1;
        this->_outChannel   = 1;
    }
    else if(this->_axis == 0)  

    {
        this->_outWidth     = width;
        this->_outHeight    = height;
        this->_outChannel   = 1;
    }
    else if(this->_axis == 1)  

    {
        this->_outWidth     = width;
        this->_outHeight    = 1;
        this->_outChannel   = channel;
    }
    else if(this->_axis == 2)  

    {
        this->_outWidth     = 1;
        this->_outHeight    = height;
        this->_outChannel   = channel;
    }

    this->_inputNum  =   width * height * channel;
    this->_outputNum =   this->_outWidth * this->_outHeight * this->_outChannel;

    this->_layerDetail.append("Reduction Layer : " + ReductionParams::getStrFromReduceType(reductionType));

    this->_maxOutputNum  = this->_batch*this->_outputNum;

    char msg[100];

#ifdef WIN32
    sprintf_s(msg, "        %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
              this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#else
    sprintf(msg, "        %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
            this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#endif
    this->_layerDetail.append(msg);

}

ReductionLayer::~ReductionLayer()
{

}

int ReductionLayer::getAxis() const
{
    return _axis;
}

void ReductionLayer::forward(NetworkState &netState)
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

    Blas::cpuFill(this->_outputNum*this->_batch,0,layerOutput,1);

    if(this->_axis != -1)
    {
        for (int b = 0; b < this->_batch; ++b)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int c = 0; c < this->_channel; ++c)
            {
                for (int h = 0; h < this->_height; ++h)
                {
                    for (int w = 0; w < this->_width; ++w)
                    {
                        if(this->_axis == 0)
                        {
#pragma omp atomic
                            layerOutput[b*this->_height*this->_width + h*this->_width + w] += layerInput[b*this->_channel*this->_height*this->_width + c*this->_height*this->_width + h*this->_width + w];
                        }
                        if(this->_axis == 1)
                        {
#pragma omp atomic
                            layerOutput[b*this->_channel*this->_width + c*this->_width + w] += layerInput[b*this->_channel*this->_height*this->_width + c*this->_height*this->_width + h*this->_width + w];
                        }
                        if(this->_axis == 2)
                        {
#pragma omp atomic
                            layerOutput[b*this->_channel*this->_height + c*this->_height + h] += layerInput[b*this->_channel*this->_height*this->_width + c*this->_height*this->_width + h*this->_width + w];
                        }
                    }
                }
            }
        }

        if(this->_reductionType == ReductionType::REDUCTION_MEAN)
        {

            if(this->_axis == 0)
            {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
                for (int i = 0; i < this->_batch * this->_outputNum; ++i)
                {
                    layerOutput[i] /= this->_channel;
                }
            }

            if(this->_axis == 1)
            {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
                for (int i = 0; i < this->_batch * this->_outputNum; ++i)
                {
                    layerOutput[i] /= this->_height;
                }
            }

            if(this->_axis == 2)
            {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
                for (int i = 0; i < this->_batch * this->_outputNum; ++i)
                {
                    layerOutput[i] /= this->_width;
                }
            }
        }
    }

    if(this->_axis == -1)
    {
        float count = 0.f;
        for (int b = 0; b < this->_batch; ++b)
        {
#ifdef USE_OMP                                  

#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:count)
#endif
            for (int k = 0; k < this->_width*this->_height*this->_channel; ++k)
            {
                count += (layerInput[k]) ;
            }
            if(this->_reductionType == ReductionType::REDUCTION_MEAN)
            {
                layerOutput[b] = count/this->_inputNum;
            }
            else
            {
                layerOutput[b] = count;
            }
        }
    }

    this->_forwardTime = TimeUtil::getElapsedTime(st);
}

void ReductionLayer::mallocMemory()
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
                this->_gpuOutput    =   Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }
    this->_memReUse         =  0;
}

ReductionType ReductionLayer::getReductionType() const
{
    return _reductionType;
}

#ifdef USE_GPU
void ReductionLayer::forwardGPU(NetworkState &netState)
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

    BlasGPU::gpuFill(this->_outputNum*this->_batch,0,layerGpuOutput,1);

    BlasGPU::gpuFastSum(this->_axis, this->_batch, this->_channel, this->_width, this->_height, layerGpuInput, layerGpuOutput, this->_reductionType);

}
#endif

}

