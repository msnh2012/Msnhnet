#include "Msnhnet/layers/MsnhDeConvolutionalLayer.h"

namespace Msnhnet
{
DeConvolutionalLayer::DeConvolutionalLayer(const int &batch, const int &height, const int &width, const int &channel, const int &num,
                                           const int &kSizeX, const int &kSizeY, const int &strideX, const int &strideY,
                                           const int &paddingX, const int &paddingY, const int &groups,
                                           const ActivationType &activation,  const std::vector<float> &actParams, const int &useBias)
{
    this->_type          =   LayerType::DECONVOLUTIONAL;
    this->_layerName     =  "DeConv          ";

    this->_height        =   height;
    this->_width         =   width;
    this->_channel       =   channel;
    this->_num           =   num;
    this->_batch         =   batch;

    this->_kSizeX        =   kSizeX;
    this->_kSizeY        =   kSizeY;

    this->_strideX       =   strideX;
    this->_strideY       =   strideY;

    this->_paddingX      =   paddingX;
    this->_paddingY      =   paddingY;

    this->_activation    =   activation;
    this->_actParams     =   actParams;

    this->_useBias       =   useBias;

    if(groups<1)
    {
        this->_groups    = 1;
    }
    else
    {
        this->_groups    = groups;
    }

    this->_outHeight     =   deConvOutHeight();
    this->_outWidth      =   deConvOutWidth();
    this->_outChannel    =   this->_num;
    this->_outputNum     =   this->_outWidth * this->_outHeight * this->_outChannel;
    this->_inputNum      =   width * width * channel;

    this->_nWeights      =   (this->_channel/this->_groups)*this->_num*this->_kSizeX*this->_kSizeY;

    if(this->_useBias)
    {
        this->_nBiases       = this->_num;
    }
    else
    {
        this->_nBiases       =   0;
    }

    this->_numWeights            =   static_cast<size_t>(this->_nWeights + this->_nBiases);

    if(!BaseLayer::isPreviewMode)
    {
        this->_weights       =   new float[static_cast<size_t>(this->_nWeights)]();
        if(this->_useBias)
        {
            this->_biases        =   new float[static_cast<size_t>(this->_nBiases)]();
        }
        this->_colImg        =   new float[static_cast<size_t>(this->_height * this->_width * this->_kSizeX * this->_kSizeX * this->_num)]();
        this->_output        =   new float[static_cast<size_t>(_outputNum * this->_batch)]();
    }
    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", this->_height, this->_width, this->_channel, this->_num,
              this->_outHeight, this->_outWidth, this->_num);
#else
    sprintf(msg, "Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", this->_height, this->_width, this->_channel, this->_num,
            this->_outHeight, this->_outWidth, this->_num);
#endif
    this->_layerDetail   = msg;

}

void DeConvolutionalLayer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();

    int mOutH           =   deConvOutHeight();
    int mOutW           =   deConvOutWidth();
    int whOutSize       =   mOutH*mOutW;

    int m               =   this->_kSizeX * this->_kSizeY * this->_num / this->_groups;
    int n               =   this->_height * this->_width;
    int k               =   this->_channel / this->_groups;

    Blas::cpuFill(this->_outputNum*this->_batch, 0, this->_output, 1);

    for (int i = 0; i < this->_batch; ++i)
    {
        for (int j = 0; j < this->_groups; ++j)
        {
            float *a        =   this->_weights + j*this->_nWeights /this->_groups;
            float *b        =   netState.input + i*this->_channel/this->_groups*this->_height*this->_width;
            float *c        =   this->_colImg + (i*this->_groups + j)*n*k;
            Gemm::cpuGemm(1,0,m,n,k,1,a,m,b,n,0,c,n, this->supportAvx&&this->supportFma);

            Gemm::cpuCol2Im(c, this->_num/this->_groups, mOutH, mOutW, this->_kSizeX, this->_kSizeY, this->_strideX, this->_strideY, this->_paddingX,
                            this->_paddingY, this->_output + i*this->_num*whOutSize);
        }

    }
    ConvolutionalLayer::addBias(this->_output, this->_biases, this->_batch, this->_num, whOutSize);

    if(this->_activation == ActivationType::NORM_CHAN)
    {
        Activations::activateArrayNormCh(this->_output, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                         this->_outWidth*this->_outHeight, this->_output);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        Activations::activateArrayNormChSoftMax(this->_output, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, this->_output,0);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        Activations::activateArrayNormChSoftMax(this->_output, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, this->_output,1);
    }
    else if(this->_activation == ActivationType::NONE)
    {

    }
    else
    {
        if(_actParams.size() > 0)
        {
            Activations::activateArray(this->_output, this->_outputNum*this->_batch, this->_activation, this->supportAvx, _actParams[0]);
        }
        else
        {
            Activations::activateArray(this->_output, this->_outputNum*this->_batch, this->_activation, this->supportAvx);
        }
    }

    this->_forwardTime =  TimeUtil::getElapsedTime(st);

}

void DeConvolutionalLayer::loadAllWeigths(std::vector<float> &weights)
{
    if(weights.size() != this->_numWeights)
    {
        throw Exception(1,"Deconv weights load err. needed : " + std::to_string(this->_numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__, __FUNCTION__);
    }

    loadWeights(weights.data(), _nWeights);

    if(_useBias==1)
    {
        loadBias(weights.data() + _nWeights, _nBiases);
    }
}

void DeConvolutionalLayer::loadBias(float * const &bias, const int &len)
{
    if(len != this->_nBiases)
    {
        throw Exception(1, "load bias data len error ",__FILE__,__LINE__, __FUNCTION__);
    }
    Blas::cpuCopy(len, bias, 1, this->_biases,1);
}

void DeConvolutionalLayer::loadWeights(float * const &weights, const int &len)
{
    if(len != this->_nWeights)
    {
        throw Exception(1, "load weights data len error ",__FILE__,__LINE__, __FUNCTION__);
    }
    Blas::cpuCopy(len, weights, 1, this->_weights,1);
}

int DeConvolutionalLayer::deConvOutHeight()
{
    return (this->_height - 1) * this->_strideY -2*_paddingY + this->_kSizeY;

}

int DeConvolutionalLayer::deConvOutWidth()
{
    return (this->_width - 1) * this->_strideX -2*_paddingX + this->_kSizeX;

}

float *DeConvolutionalLayer::getWeights() const
{
    return _weights;
}

float *DeConvolutionalLayer::getBiases() const
{
    return _biases;
}

float *DeConvolutionalLayer::getColImg() const
{
    return _colImg;
}

int DeConvolutionalLayer::getKSizeX() const
{
    return _kSizeX;
}

int DeConvolutionalLayer::getKSizeY() const
{
    return _kSizeY;
}

int DeConvolutionalLayer::getStride() const
{
    return _stride;
}

int DeConvolutionalLayer::getStrideX() const
{
    return _strideX;
}

int DeConvolutionalLayer::getStrideY() const
{
    return _strideY;
}

int DeConvolutionalLayer::getPaddingX() const
{
    return _paddingX;
}

int DeConvolutionalLayer::getPaddingY() const
{
    return _paddingY;
}

int DeConvolutionalLayer::getUseBias() const
{
    return _useBias;
}

int DeConvolutionalLayer::getNBiases() const
{
    return _nBiases;
}

int DeConvolutionalLayer::getNWeights() const
{
    return _nWeights;
}

int DeConvolutionalLayer::getGroups() const
{
    return _groups;
}

}
