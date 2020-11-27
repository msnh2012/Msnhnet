#include "Msnhnet/layers/MsnhSliceLayer.h"
namespace Msnhnet
{

SliceLayer::SliceLayer(const int &batch, const int &height, const int &width, const int &channel, const int &start0, const int &step0,
                       const int &start1, const int &step1, const int &start2, const int &step2)
{
    this->_batch        = batch;
    this->_height       = height;
    this->_width        = width;
    this->_channel      = channel;

    this->_layerName    =  "Slice           ";
    this->_type         =   LayerType::SLICE;

    this->_start0       =   start0;
    this->_step0        =   step0;
    this->_start1       =   start1;
    this->_step1        =   step1;
    this->_start2       =   start2;
    this->_step2        =   step2;

    this->_outChannel   =   this->_channel;
    this->_outWidth     =   this->_width;
    this->_outHeight    =   this->_height;

    if(this->_start0 >(this->_channel - 1))
    {
        throw Exception(1,"slice c start error",__FILE__,__LINE__,__FUNCTION__);
    }

    int chTmp          =  ((this->_channel - this->_start0)%this->_step0)==0?0:1;
    this->_outChannel = (this->_channel - this->_start0)/this->_step0 + chTmp;

    if(this->_start1 >(this->_height - 1))
    {
        throw Exception(1,"slice h start error",__FILE__,__LINE__,__FUNCTION__);
    }

    int hTmp            =  ((this->_height - this->_start1)%this->_step1)==0?0:1;
    this->_outHeight    =  (this->_height - this->_start1)/this->_step1 + hTmp;

    if(this->_start2 >(this->_width - 1))
    {
        throw Exception(1,"slice w start error",__FILE__,__LINE__,__FUNCTION__);
    }

    int wTmp           =  ((this->_width - this->_start2)%this->_step2)==0?0:1;
    this->_outWidth    =  (this->_width - this->_start2)/this->_step2 + wTmp;

    this->_inputNum     =   this->_width * this->_height * this->_channel;
    this->_outputNum    =   this->_outWidth * this->_outHeight * this->_outChannel;

    this->_maxOutputNum = this->_batch*this->_outputNum;

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Slice                        %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
              this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#else
    sprintf(msg, "Slice                        %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
            this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#endif
    this->_layerDetail = msg;
}

void SliceLayer::mallocMemory()
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

void SliceLayer::forward(NetworkState &netState)
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
                    layerOutput[b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + h*this->_outWidth + w]
                            =
                    layerInput[b*this->_channel*this->_height*this->_width + (c*this->_step0 + this->_start0)*this->_height*this->_width +
                                                                             (h*this->_step1 + this->_start1)*this->_width +
                                                                             (w*this->_step2 + this->_start2)];
                }
            }
        }
    }

    this->_forwardTime = TimeUtil::getElapsedTime(st);
    return;
}

int SliceLayer::getStart0() const
{
    return _start0;
}

int SliceLayer::getStep0() const
{
    return _step0;
}

int SliceLayer::getStart1() const
{
    return _start1;
}

int SliceLayer::getStep1() const
{
    return _step1;
}

int SliceLayer::getStart2() const
{
    return _start2;
}

int SliceLayer::getStep2() const
{
    return _step2;
}

#ifdef USE_GPU
void SliceLayer::forwardGPU(NetworkState &netState)
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

    SliceLayerGPU::forwardNormalGPU(this->_batch, this->_outChannel, this->_outHeight, this->_outWidth, this->_height, this->_width, this->_channel,
                                    this->_start0, this->_step0, this->_start1, this->_step1, this->_start2, this->_step2, layerGpuInput, layerGpuOutput);
    this->recordCudaStop();
}
#endif

}

