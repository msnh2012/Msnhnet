#include "Msnhnet/layers/MsnhDeConvolutionalLayer.h"

namespace Msnhnet
{
DeConvolutionalLayer::DeConvolutionalLayer(const int &batch, const int &height, const int &width, const int &channel, const int &num,
                                           const int &kSizeX, const int &kSizeY, const int &strideX, const int &strideY,
                                           const int &paddingX, const int &paddingY,
                                           const ActivationType &activation,  const std::vector<float> &actParams, const int &useBias)
{
    this->type          =   LayerType::DECONVOLUTIONAL;
    this->layerName     =  "DeConv          ";

    this->height        =   height;
    this->width         =   width;
    this->channel       =   channel;
    this->num           =   num;
    this->batch         =   batch;

    this->kSizeX        =   kSizeX;
    this->kSizeY        =   kSizeY;

    this->strideX       =   strideX;
    this->strideY       =   strideY;

    this->paddingX      =   paddingX;
    this->paddingY      =   paddingY;

    this->activation    =   activation;
    this->actParams     =   actParams;

    this->useBias       =   useBias;

    this->outHeight     =   deConvOutHeight();
    this->outWidth      =   deConvOutWidth();
    this->outChannel    =   this->num;
    this->outputNum     =   this->outWidth * this->outHeight * this->outChannel;
    this->inputNum      =   width * width * channel;

    this->nWeights      =   this->channel*this->num*this->kSizeX*this->kSizeY;

    if(this->useBias)
    {
        this->nBiases       = this->num;
    }
    else
    {
        this->nBiases       =   0;
    }

    this->numWeights            =   static_cast<size_t>(this->nWeights + this->nBiases);

    if(!BaseLayer::isPreviewMode)
    {
        this->weights       =   new float[static_cast<size_t>(this->nWeights)]();
        if(this->useBias)
        {
            this->biases        =   new float[static_cast<size_t>(this->nBiases)]();
        }
        this->colImg        =   new float[static_cast<size_t>(this->height * this->width * this->kSizeX * this->kSizeX * this->num)]();
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

    int m               =   this->kSizeX * this->kSizeY * this->num;
    int n               =   this->height * this->width;
    int k               =   this->channel;

    Blas::cpuFill(this->outputNum*this->batch, 0, this->output, 1);

    for (int i = 0; i < this->batch; ++i)
    {
        float *a        =   this->weights;
        float *b        =   netState.input + i*this->channel*this->height*this->width;
        float *c        =   this->colImg;
        Gemm::cpuGemm(1,0,m,n,k,1,a,m,b,n,0,c,n, this->supportAvx&&this->supportFma);

        Gemm::cpuCol2Im(c, this->num, mOutH, mOutW, this->kSizeX, this->kSizeY, this->strideX, this->strideY, this->paddingX,
                        this->paddingY, this->output + i*this->num*whOutSize);
    }
    ConvolutionalLayer::addBias(this->output, this->biases, this->batch, this->num, whOutSize);

    if(this->activation == ActivationType::NORM_CHAN)
    {
        Activations::activateArrayNormCh(this->output, this->outputNum*this->batch, this->batch, this->outChannel,
                                         this->outWidth*this->outHeight, this->output);
    }
    else if(this->activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        Activations::activateArrayNormChSoftMax(this->output, this->outputNum*this->batch, this->batch, this->outChannel,
                                                this->outWidth*this->outHeight, this->output,0);
    }
    else if(this->activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        Activations::activateArrayNormChSoftMax(this->output, this->outputNum*this->batch, this->batch, this->outChannel,
                                                this->outWidth*this->outHeight, this->output,1);
    }
    else if(this->activation == ActivationType::NONE)
    {

    }
    else
    {
        if(actParams.size() > 0)
        {
            Activations::activateArray(this->output, this->outputNum*this->batch, this->activation, this->supportAvx, actParams[0]);
        }
        else
        {
            Activations::activateArray(this->output, this->outputNum*this->batch, this->activation, this->supportAvx);
        }
    }

    auto so = std::chrono::system_clock::now();
    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

void DeConvolutionalLayer::loadAllWeigths(std::vector<float> &weights)
{
    if(weights.size() != this->numWeights)
    {
        throw Exception(1,"Deconv weights load err. needed : " + std::to_string(this->numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__);
    }

    loadWeights(weights.data(), nWeights);

    if(useBias==1)
    {
        loadBias(weights.data() + nWeights, nBiases);
    }
}

void DeConvolutionalLayer::loadBias(float * const &bias, const int &len)
{
    if(len != this->nBiases)
    {
        throw Exception(1, "load bias data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, bias, 1, this->biases,1);
}

void DeConvolutionalLayer::loadWeights(float * const &weights, const int &len)
{
    if(len != this->nWeights)
    {
        throw Exception(1, "load weights data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, weights, 1, this->weights,1);
}

int DeConvolutionalLayer::deConvOutHeight()
{
    return (this->height - 1) * this->strideY -2*paddingY + this->kSizeY;

}

int DeConvolutionalLayer::deConvOutWidth()
{
    return (this->width - 1) * this->strideX -2*paddingX + this->kSizeX;

}

}
