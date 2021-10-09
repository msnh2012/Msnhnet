#include "Msnhnet/layers/MsnhClipLayer.h"
#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhClipLayerGPU.h"
#endif

namespace Msnhnet
{

ClipLayer::ClipLayer(const int &batch, const int &height, const int &width, const int &channel, const float &min, const float &max)
{
    this->_batch        = batch;
    this->_height       = height;
    this->_width        = width;
    this->_channel      = channel;

    this->_layerName    =  "Clip            ";
    this->_type         =  LayerType::CLIP;

    this->_min          = min;
    this->_max          = max;

    if(this->_min > this->_max)
    {
        throw Exception(1,"Clip min val must < max val [min:max]=[" + std::to_string(this->_min)+":"+std::to_string(this->_max)+"]\n",__FILE__, __LINE__, __FUNCTION__);
    }

    this->_outChannel   =   this->_channel;
    this->_outWidth     =   this->_width;
    this->_outHeight    =   this->_height;

    this->_inputNum     =   this->_width * this->_height * this->_channel;
    this->_outputNum    =   this->_outWidth * this->_outHeight * this->_outChannel;

    this->_maxOutputNum =   this->_batch*this->_outputNum;

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Slice                        min = %4f, max = %4f \n", this->_min, this->_max);
#else
    sprintf(msg, "Slice                        min = %4f, max = %4f \n", this->_min, this->_max);
#endif
    this->_layerDetail = msg;
}

void ClipLayer::mallocMemory()
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
                this->_gpuOutput =  Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }
    this->_memReUse         =  0;
}

void ClipLayer::forward(NetworkState &netState)
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
                    int idx = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + h*this->_outWidth + w;
                    if(layerInput[idx]>this->_max)
                        layerOutput[idx] = this->_max;
                    else if(layerInput[idx]<this->_min)
                        layerOutput[idx] = this->_min;
                    else
                        layerOutput[idx] = layerInput[idx];
                }
            }
        }
    }

    this->_forwardTime = TimeUtil::getElapsedTime(st);
    return;
}

#ifdef USE_GPU
void ClipLayer::forwardGPU(NetworkState &netState)
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

    ClipLayerGPU::forwardNormalGPU(this->_batch, this->_outChannel, this->_outHeight, this->_outWidth,
                                      this->_min, this->_max,
                                      layerGpuInput, layerGpuOutput
                                      );
    this->recordCudaStop();
}
#endif

float ClipLayer::getMax() const
{
    return _max;
}

float ClipLayer::getMin() const
{
    return _min;
}

}

