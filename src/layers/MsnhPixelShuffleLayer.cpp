#include "Msnhnet/layers/MsnhPixelShuffLeLayer.h"
namespace Msnhnet
{

PixelShuffleLayer::PixelShuffleLayer(const int &batch, const int &height, const int &width, const int &channel, const int &factor)
{
    this->_batch    =   batch;
    this->_channel  =   channel;
    this->_height   =   height;
    this->_width    =   width;

    this->_layerName =  "PixelShuffle    ";
    this->_type      =   LayerType::PIXEL_SHUFFLE;

    if(factor<=0)
    {
        throw Exception(1,"factor must > 0",__FILE__, __LINE__, __FUNCTION__);
    }

    if(channel%(factor*factor)!=0)
    {
        throw Exception(1,"channel is not divisible by factor^2",__FILE__, __LINE__, __FUNCTION__);
    }

    this->_factor       =  factor;
    this->_outWidth     =  width*factor;
    this->_outHeight    =  height*factor;
    this->_outChannel   =  channel/factor/factor;

    this->_inputNum  =   width * height * channel;
    this->_outputNum =   this->_outWidth * this->_outHeight * this->_outChannel;

    this->_maxOutputNum  = this->_batch*this->_outputNum;

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "PixelShuffle                 %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
              this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#else
    sprintf(msg, "PixelShuffle                 %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
            this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#endif
    this->_layerDetail = msg;
}

PixelShuffleLayer::~PixelShuffleLayer()
{

}

void PixelShuffleLayer::forward(NetworkState &netState)
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

    for (int b = 0; b < this->_batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int c = 0; c < this->_outChannel; ++c)
        {
            for (int h = 0; h < this->_outHeight; ++h)
            {
                for (int w = 0; w < this->_outWidth; ++w)
                {
                    layerOutput[b*this->_outChannel*this->_outWidth*this->_outHeight + c*this->_outWidth*this->_outHeight + h*this->_outWidth + w]
                            =
                    layerInput[b*this->_channel*this->_width*this->_height +
                               c*this->_factor*this->_width*this->_height +
                               h*this->_width +
                               w%(this->_factor)*this->_width*this->_height +
                               w/this->_factor ];
                }
            }
        }
    }

    this->_forwardTime = TimeUtil::getElapsedTime(st);
}

void PixelShuffleLayer::mallocMemory()
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
                this->_gpuOutput =   Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }
    this->_memReUse         =  0;
}

#ifdef USE_GPU
void PixelShuffleLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

    float* layerGpuInput   = netState.getGpuOutput();
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

    PixelShuffleLayerGPU::forwardNormalGPU(this->_batch, this->_outChannel, this->_outHeight, this->_outWidth,
                                      this->_height, this->_width, this->_channel,
                                      this->_factor,layerGpuInput, layerGpuOutput
                                      );
    this->recordCudaStop();
}
#endif

int PixelShuffleLayer::getFactor() const
{
    return _factor;
}

}
