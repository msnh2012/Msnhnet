#include "Msnhnet/layers/MsnhUpSampleLayer.h"
namespace Msnhnet
{
UpSampleLayer::UpSampleLayer(const int &batch, const int &width, const int &height, const int &channel, const int &stride, const float &scale)
{
    this->type          =   LayerType::UPSAMPLE;
    this->layerName     =   "UpSample        ";

    this->batch         =   batch;
    this->width         =   width;
    this->height        =   height;
    this->channel       =   channel;

    this->outWidth      =   width*stride;
    this->outHeight     =   height*stride;
    this->outChannel    =   channel;

    this->scale         =   scale;

    int mStride         =   stride;

    if(stride < 0)
    {
        mStride         =   -stride;
        this->reverse   =   1;
        this->outWidth  =   width/mStride;
        this->outHeight =   height/mStride;
    }
    this->stride        =   mStride;
    this->outputNum     =   this->outWidth * this->outHeight * this->outChannel;
    this->inputNum      =   this->width * this->height  * this->channel;

    if(!BaseLayer::isPreviewMode)
    {
        this->output        =   new float[static_cast<size_t>(this->outputNum * this->batch)]();
    }

    char msg[100];
    if(this->reverse)
    {
#ifdef WIN32
        sprintf_s(msg, "downsample              %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->stride, this->width, this->height, this->channel,
                  this->outHeight, this->outHeight, this->outChannel);
#else
        sprintf(msg, "downsample              %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->stride, this->width, this->height, this->channel,
                this->outHeight, this->outHeight, this->outChannel);
#endif
    }
    else
    {
#ifdef WIN32
        sprintf_s(msg, "upsample                %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->stride, this->width, this->height, this->channel,
                  this->outHeight, this->outHeight, this->outChannel);
#else
        sprintf(msg, "upsample                %2dx  %4d x%4d x%4d -> %4d x%4d x%4d\n", this->stride, this->width, this->height, this->channel,
                this->outHeight, this->outHeight, this->outChannel);
#endif
    }

    this->layerDetail   = msg;
}

void UpSampleLayer::forward(NetworkState &netState)
{

    auto st = std::chrono::system_clock::now();

    if(this->reverse)
    {
        Blas::cpuUpSample(this->output, this->outWidth, this->outHeight, this->channel, this->batch, this->stride, 0, this->scale, netState.input);
    }
    else
    {
        Blas::cpuUpSample(netState.input, this->width, this->height, this->channel, this->batch, this->stride, 1, this->scale, this->output);
    }

    auto so = std::chrono::system_clock::now();
    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

void UpSampleLayer::resize(const int &width, const int &height)
{
    this->width         =   width;
    this->height        =   height;
    this->outWidth      =   width*this->stride;
    this->outHeight     =   height*this->stride;

    if(this->reverse)
    {
        this->outWidth  =   width/this->stride;
        this->outHeight =   height/this->stride;
    }

    this->outputNum     =   this->outWidth * this->outHeight * this->outChannel;

    if(this->output == nullptr)
    {
        throw Exception(1,"output can't be null", __FILE__, __LINE__);
    }

    this->output    = static_cast<float *>(realloc(this->output, static_cast<size_t>(this->outputNum * this->batch) *sizeof(float)));
}
}
