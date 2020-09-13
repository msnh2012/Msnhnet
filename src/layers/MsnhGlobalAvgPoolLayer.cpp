#include "Msnhnet/layers/MsnhGlobalAvgPoolLayer.h"
namespace Msnhnet
{
GlobalAvgPoolLayer::GlobalAvgPoolLayer(const int &batch, const int &height, const int &width, const int &channel)
{
    this->_type              = LayerType::GLOBAL_AVGPOOL;

    this->_layerName         = "GlobalAvgPool   ";

    this->_batch             = batch;
    this->_height            = height;
    this->_width             = width;
    this->_channel           = channel;

    this->_outWidth          = 1;
    this->_outHeight         = 1;
    this->_outChannel        = channel;
    this->_inputNum          = height*width*channel;
    this->_outputNum         = this->_outChannel;

    this->_bFlops            = (this->_width*this->_height* this->_channel*this->_outHeight*this->_outWidth)/ 1000000000.f;

    this->_maxOutputNum  = this->_batch*this->_outputNum;

    char msg[100];

#ifdef WIN32
    sprintf_s(msg, "GlobalAvgPool                %4d x%4d x%4d ->   %4d\n %5.3f BF\n",
              this->_width, this->_height, this->_channel, this->_outChannel, this->_bFlops);
#else
    sprintf(msg, "GlobalAvgPool                %4d x%4d x%4d ->   %4d\n %5.3f BF\n",
            this->_width, this->_height, this->_channel, this->_outChannel, this->_bFlops);
#endif
}

GlobalAvgPoolLayer::~GlobalAvgPoolLayer()
{

}

void GlobalAvgPoolLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu) 

            {
                this->_output         = new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
            }
#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuOutput      = Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }
    this->_memReUse         =  0;
}

#ifdef USE_GPU
void GlobalAvgPoolLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

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

    GlobalAvgPoolLayerGPU::forwardNormalGPU(this->_width, this->_height, this->_channel, this->_batch, layerGpuInput, layerGpuOutput);
    this->recordCudaStop();
}
#endif

void GlobalAvgPoolLayer::forward(NetworkState &netState)
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
        for (int k = 0; k < this->_channel; ++k)
        {
            int outIndex = k + b*this->_channel;
            layerOutput[outIndex] = 0;
            for (int i = 0; i < this->_height*this->_width; ++i)
            {
                int inIndex = i + this->_height*this->_width*(k + b*this->_channel);
                layerOutput[outIndex] += layerInput[inIndex];
            }
            layerOutput[outIndex] /= (this->_height*this->_width);
        }
    }

    this->_forwardTime =  TimeUtil::getElapsedTime(st);
}

}

