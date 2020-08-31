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

    if(!BaseLayer::isPreviewMode)
    {
        this->_output        =   new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
#ifdef USE_GPU
        this->_gpuOutput         = Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
#endif
    }

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

    if(this->_upsampleType == UpSampleParams::NEAREST)
    {
        Blas::cpuUpSample(netState.input, this->_width, this->_height, this->_channel, this->_batch, this->_strideX, this->_strideY, this->_scaleX, this->_output);
    }
    else
    {
        Blas::cpuBilinearResize(netState.input, this->_width, this->_height, this->_channel, this->_batch, this->_outWidth, this->_outHeight, this->_alignCorners, this->_output);
    }

    this->_forwardTime =   TimeUtil::getElapsedTime(st);

}

#ifdef USE_GPU
void UpSampleLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

    BlasGPU::gpuFill(this->_outputNum, 0, this->_gpuOutput, 1);

    if(this->_upsampleType == UpSampleParams::NEAREST)
    {
        BlasGPU::gpuUpSample(netState.input, this->_width, this->_height, this->_channel, this->_batch, this->_strideX, this->_strideY, this->_scaleX, this->_gpuOutput);
    }
    else
    {
        BlasGPU::gpuBilinearResize(netState.input, this->_width, this->_height, this->_channel, this->_batch, this->_outWidth, this->_outHeight, this->_alignCorners, this->_gpuOutput);
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
