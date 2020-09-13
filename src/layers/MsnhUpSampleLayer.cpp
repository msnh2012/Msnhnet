#include "Msnhnet/layers/MsnhUpSampleLayer.h"
namespace Msnhnet
{
UpSampleLayer::UpSampleLayer(const int &batch, const int &width, const int &height, const int &channel, const int &strideX, const int &strideY,
                             const float &scaleX, const float &scaleY, UpSampleParams::UpsampleType upsampleType, const int &alignCorners)
{
    this->_type          =   LayerType::UPSAMPLE;
    this->_layerName     =   "UpSample        ";

    this->_batch         =   batch;
    this->_width         =   width;
    this->_height        =   height;
    this->_channel       =   channel;

    this->_upsampleType  =   upsampleType;

    if(upsampleType == UpSampleParams::NEAREST)
    {
        this->_outWidth  =   width*strideX;
        this->_outHeight =   height*strideY;
    }
    else if(upsampleType == UpSampleParams::BILINEAR)
    {
        this->_outWidth  =   static_cast<int>(width*scaleX);
        this->_outHeight =   static_cast<int>(height*scaleY);
    }

    this->_outChannel    =   channel;

    this->_scaleX        =   scaleX;
    this->_scaleY        =   scaleY;

    this->_strideX       =   strideX;
    this->_strideY       =   strideY;

    this->_outputNum     =   this->_outWidth * this->_outHeight * this->_outChannel;
    this->_inputNum      =   this->_width * this->_height  * this->_channel;

    this->_alignCorners  = alignCorners;

    this->_maxOutputNum  = this->_batch*this->_outputNum;

    char msg[100];
    if(upsampleType == UpSampleParams::NEAREST)
    {
#ifdef WIN32
        sprintf_s(msg, "Upsample nearest      %2dx%2d  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->_strideX, this->_strideY, this->_width, this->_height, this->_channel,
                  this->_outHeight, this->_outHeight, this->_outChannel);
#else
        sprintf(msg, "upsample              %2dx%2d  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->_strideX, this->_strideY, this->_width, this->_height, this->_channel,
                this->_outHeight, this->_outHeight, this->_outChannel);
#endif
    }
    else if(upsampleType == UpSampleParams::BILINEAR)
    {
#ifdef WIN32
        sprintf_s(msg, "Upsample bili %4fx%4f  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->_scaleX, this->_scaleY, this->_width, this->_height, this->_channel,
                  this->_outHeight, this->_outHeight, this->_outChannel);
#else
        sprintf(msg, "upsample bili %4fx%4f  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->_scaleX, this->_scaleY, this->_width, this->_height, this->_channel,
                this->_outHeight, this->_outHeight, this->_outChannel);
#endif
    }

    this->_layerDetail   = msg;
}

void UpSampleLayer::forward(NetworkState &netState)
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

    if(this->_upsampleType == UpSampleParams::NEAREST)
    {
        Blas::cpuUpSample(layerInput, this->_width, this->_height, this->_channel, this->_batch, this->_strideX, this->_strideY, this->_scaleX, layerOutput);
    }
    else
    {
        Blas::cpuBilinearResize(layerInput, this->_width, this->_height, this->_channel, this->_batch, this->_outWidth, this->_outHeight, this->_alignCorners, layerOutput);
    }

    this->_forwardTime =   TimeUtil::getElapsedTime(st);

}

void UpSampleLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu) 

            {
                this->_output        =   new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
            }
#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuOutput     =   Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }
    this->_memReUse         =  0;
}

#ifdef USE_GPU
void UpSampleLayer::forwardGPU(NetworkState &netState)
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

    this->recordCudaStart();

    BlasGPU::gpuFill(this->_outputNum*this->_batch, 0, layerGpuOutput, 1);

    if(this->_upsampleType == UpSampleParams::NEAREST)
    {
        BlasGPU::gpuUpSample(layerGpuInput, this->_width, this->_height, this->_channel, this->_batch, this->_strideX, this->_strideY, this->_scaleX, layerGpuOutput);
    }
    else
    {
        BlasGPU::gpuBilinearResize(layerGpuInput, this->_width, this->_height, this->_channel, this->_batch, this->_outWidth, this->_outHeight, this->_alignCorners, layerGpuOutput);
    }

    this->recordCudaStop();
}
#endif

void UpSampleLayer::resize(const int &width, const int &height)
{
    this->_width         =   width;
    this->_height        =   height;
    this->_outWidth      =   width*this->_strideX;
    this->_outHeight     =   height*this->_strideY;

    this->_outputNum     =   this->_outWidth * this->_outHeight * this->_outChannel;

    if(this->_output == nullptr)
    {
        throw Exception(1,"output can't be null", __FILE__, __LINE__, __FUNCTION__);
    }

    this->_output    = static_cast<float *>(realloc(this->_output, static_cast<size_t>(this->_outputNum * this->_batch) *sizeof(float)));
}

int UpSampleLayer::getStrideX() const
{
    return _strideX;
}

int UpSampleLayer::getStrideY() const
{
    return _strideY;
}

float UpSampleLayer::getScaleX() const
{
    return _scaleX;
}

float UpSampleLayer::getScaleY() const
{
    return _scaleY;
}

int UpSampleLayer::getAlignCorners() const
{
    return _alignCorners;
}

UpSampleParams::UpsampleType UpSampleLayer::getUpsampleType() const
{
    return _upsampleType;
}
}
