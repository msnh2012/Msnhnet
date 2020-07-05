#include "Msnhnet/layers/MsnhDeConvolutionalLayer.h"

namespace Msnhnet
{
DeConvolutionalLayer::DeConvolutionalLayer(const int &batch, const int &height, const int &channel,
                                           const int &num, const int &kSize, const int &stride,
                                           const ActivationType &activation,  const std::vector<float> &actParams)
{
    this->type          =   LayerType::DECONVOLUTIONAL;
    this->layerName     =  "DeConv          ";

    this->height        =   height;
    this->width         =   width;
    this->channel       =   channel;
    this->num           =   num;
    this->batch         =   batch;
    this->stride        =   stride;
    this->kSize         =   kSize;

    this->activation    =   activation;
    this->actParams     =   actParams;

    this->outHeight     =   deConvOutHeight();
    this->outWidth      =   deConvOutWidth();
    this->outChannel    =   this->num;
    this->outputNum     =   this->outWidth * this->outHeight * this->outChannel;
    this->inputNum      =   width * height * channel;

    if(!BaseLayer::isPreviewMode)
    {
        this->weights       =   new float[static_cast<size_t>(this->channel*this->num*this->kSize*this->kSize)]();
        this->biases        =   new float[static_cast<size_t>(this->num)]();
        this->colImg        =   new float[static_cast<size_t>(this->height * this->width * this->kSize * this->kSize * this->num)]();
        this->output        =   new float[static_cast<size_t>(outputNum * this->batch)]();
    }
    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", this->height, this->width, this->channel, this->num,
              this->outHeight, this->outWidth, this->num);
#else
    sprintf(msg, "Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", this->height, this->width, this->channel, this->num,
            this->outHeight, this->outWidth, this->num);
#endif
    this->layerDetail   = msg;

}

void DeConvolutionalLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    int mOutH           =   deConvOutHeight();
    int mOutW           =   deConvOutWidth();
    int whOutSize       =   mOutH*mOutW;

    int m               =   this->kSize * this->kSize * this->num;
    int n               =   this->height * this->width;
    int k               =   this->channel;

    Blas::cpuFill(this->outputNum*this->batch, 0, this->output, 1);

    for (int i = 0; i < this->batch; ++i)
    {
        float *a        =   this->weights;
        float *b        =   netState.input + i*this->channel*this->height*this->width;
        float *c        =   this->colImg;
        Gemm::cpuGemm(1,0,m,n,k,1,a,m,b,n,0,c,n, this->supportAvx&&this->supportFma);

        Gemm::cpuIm2col(c, this->num, mOutH, mOutW, this->kSize, this->stride,0,
                        this->output + i*this->num*whOutSize);
    }
    ConvolutionalLayer::addBias(this->output, this->biases, this->batch, this->num, whOutSize);

    if(actParams.size() > 0)
    {
        Activations::activateArray(this->output, this->outputNum*this->batch, this->activation, actParams[0]);
    }
    else
    {
        Activations::activateArray(this->output, this->outputNum*this->batch, this->activation);
    }

    auto so = std::chrono::system_clock::now();
    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

int DeConvolutionalLayer::deConvOutHeight()
{
    int h   =   this->stride*(this->height - 1) + this->kSize;
    return h;
}

int DeConvolutionalLayer::deConvOutWidth()
{
    int w   =   this->stride*(this->width - 1) + this->kSize;
    return w;
}
}
