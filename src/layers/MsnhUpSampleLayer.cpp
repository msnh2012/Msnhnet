#include "Msnhnet/layers/MsnhUpSampleLayer.h"
namespace Msnhnet
{
UpSampleLayer::UpSampleLayer(const int &batch, const int &width, const int &height, const int &channel, const int &stride, const float &scale)
{
    this->_type          =   LayerType::UPSAMPLE;
    this->_layerName     =   "UpSample        ";

    this->_batch         =   batch;
    this->_width         =   width;
    this->_height        =   height;
    this->_channel       =   channel;

    this->_outWidth      =   width*stride;
    this->_outHeight     =   height*stride;
    this->_outChannel    =   channel;

    this->_scale         =   scale;

    int mStride         =   stride;

    if(stride < 0)
    {
        mStride         =   -stride;
        this->_reverse   =   1;
        this->_outWidth  =   width/mStride;
        this->_outHeight =   height/mStride;
    }
    this->_stride        =   mStride;
    this->_outputNum     =   this->_outWidth * this->_outHeight * this->_outChannel;
    this->_inputNum      =   this->_width * this->_height  * this->_channel;

    if(!BaseLayer::isPreviewMode)
    {
        this->_output        =   new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
#ifdef USE_GPU
        this->_gpuOutput         = Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
#endif
    }

    char msg[100];
    if(this->_reverse)
    {
#ifdef WIN32
        sprintf_s(msg, "downsample              %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->_stride, this->_width, this->_height, this->_channel,
                  this->_outHeight, this->_outHeight, this->_outChannel);
#else
        sprintf(msg, "downsample              %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->_stride, this->_width, this->_height, this->_channel,
                this->_outHeight, this->_outHeight, this->_outChannel);
#endif
    }
    else
    {
#ifdef WIN32
        sprintf_s(msg, "upsample                %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->_stride, this->_width, this->_height, this->_channel,
                  this->_outHeight, this->_outHeight, this->_outChannel);
#else
        sprintf(msg, "upsample                %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->_stride, this->_width, this->_height, this->_channel,
                this->_outHeight, this->_outHeight, this->_outChannel);
#endif
    }

    this->_layerDetail   = msg;
}

void UpSampleLayer::forward(NetworkState &netState)
{
    TimeUtil::startRecord();

    if(this->_reverse)
    {
        Blas::cpuUpSample(this->_output, this->_outWidth, this->_outHeight, this->_channel, this->_batch, this->_stride, 0, this->_scale, netState.input);
    }
    else
    {
        Blas::cpuUpSample(netState.input, this->_width, this->_height, this->_channel, this->_batch, this->_stride, 1, this->_scale, this->_output);
    }

    this->_forwardTime =   TimeUtil::getElapsedTime();

}

#ifdef USE_GPU
void UpSampleLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

    BlasGPU::gpuFill(this->_outputNum, 0, this->_gpuOutput, 1);

    if(this->_reverse)
    {
        BlasGPU::gpuUpSample(this->_gpuOutput, this->_outWidth, this->_outHeight, this->_channel, this->_batch, this->_stride, 0, this->_scale, netState.input);
    }
    else
    {
        BlasGPU::gpuUpSample(netState.input, this->_width, this->_height, this->_channel, this->_batch, this->_stride, 1, this->_scale, this->_gpuOutput);
    }

    this->recordCudaStop();
}
#endif

void UpSampleLayer::resize(const int &width, const int &height)
{
    this->_width         =   width;
    this->_height        =   height;
    this->_outWidth      =   width*this->_stride;
    this->_outHeight     =   height*this->_stride;

    if(this->_reverse)
    {
        this->_outWidth  =   width/this->_stride;
        this->_outHeight =   height/this->_stride;
    }

    this->_outputNum     =   this->_outWidth * this->_outHeight * this->_outChannel;

    if(this->_output == nullptr)
    {
        throw Exception(1,"output can't be null", __FILE__, __LINE__, __FUNCTION__);
    }

    this->_output    = static_cast<float *>(realloc(this->_output, static_cast<size_t>(this->_outputNum * this->_batch) *sizeof(float)));
}

int UpSampleLayer::getReverse() const
{
    return _reverse;
}

int UpSampleLayer::getStride() const
{
    return _stride;
}

float UpSampleLayer::getScale() const
{
    return _scale;
}
}
