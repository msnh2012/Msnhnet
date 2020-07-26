#include "Msnhnet/layers/MsnhConnectedLayer.h"
namespace Msnhnet
{
ConnectedLayer::ConnectedLayer(const int &batch, const int &steps, const int &inputNum,
                               const int &outputNum, const ActivationType &activation, const std::vector<float> &actParams, const int &batchNorm)
{
    int totalBatch      =   batch*steps;
    this->_type          =   LayerType::CONNECTED;
    this->_layerName     =  "Connected       ";

    this->_inputNum      =   inputNum;
    this->_outputNum     =   outputNum;
    this->_batch         =   batch;
    this->_batchNorm     =   batchNorm;

    this->_height        =   1;
    this->_width         =   1;
    this->_channel       =   inputNum;

    this->_outHeight     =   1;
    this->_outWidth      =   1;
    this->_outChannel    =   outputNum;

    this->_num           =   this->_outChannel;
    this->_kSize         =   1;
    this->_stride        =   1;
    this->_padding       =   0;

    this->_activation    =   activation;
    this->_actParams     =   actParams;

    this->_nWeights      =   inputNum * outputNum;
    this->_nBiases       =   outputNum;

    if(!BaseLayer::isPreviewMode)
    {
        this->_output        =   new float[static_cast<size_t>(totalBatch * outputNum) ]();
        this->_weights       =   new float[static_cast<size_t>(inputNum * outputNum)]();
        this->_biases        =   new float[static_cast<size_t>(outputNum)]();
    }

    if(batchNorm)
    {
        if(!BaseLayer::isPreviewMode)
        {
            this->_scales        =   new float[static_cast<size_t>(outputNum)]();

            for (int i = 0; i < outputNum; ++i)
            {
                this->_scales[i] =   1;
            }

            this->_rollMean      =   new float[static_cast<size_t>(outputNum)]();
            this->_rollVariance  =   new float[static_cast<size_t>(outputNum)]();
        }

        this->_nScales       =   outputNum;
        this->_nRollMean     =   outputNum;
        this->_nRollVariance =   outputNum;
    }

    this->_numWeights            =   static_cast<size_t>(this->_nWeights + this->_nScales + this->_nRollMean + this->_nRollVariance + this->_nBiases);

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "connected                            %4d  ->  %4d\n", inputNum, outputNum);
#else
    sprintf(msg, "connected                            %4d  ->  %4d\n", inputNum, outputNum);
#endif
    this->_layerDetail       = msg;
}

ConnectedLayer::~ConnectedLayer()
{
    releaseArr(_weights);
    releaseArr(_biases);
    releaseArr(_scales);
    releaseArr(_rollMean);
    releaseArr(_rollVariance);

}

void ConnectedLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    Blas::cpuFill(this->_outputNum * this->_batch, 0, this->_output, 1);
    int m       =   this->_batch;
    int k       =   this->_inputNum;
    int n       =   this->_outputNum;

    float *a    =   netState.input;
    float *b    =   this->_weights;
    float *c    =   this->_output;

    Gemm::cpuGemm(0,1,m,n,k,1,a,k,b,k,1,c,n,this->supportAvx&&this->supportFma);

    if(this->_batchNorm == 1)
    {

        Blas::cpuNorm(this->_output, this->_rollMean, this->_rollVariance, this->_batch, this->_outputNum, 1);

        ConvolutionalLayer::scaleBias(this->_output, this->_scales, this->_batch, this->_outputNum, 1);

    }

    for (int i = 0; i < this->_batch; ++i)
    {
        Blas::cpuAxpy(this->_outputNum, 1, this->_biases, 1, this->_output + i * this->_outputNum, 1);
    }

    if(     this->_activation==ActivationType::NORM_CHAN||
            this->_activation==ActivationType::NORM_CHAN_SOFTMAX||
            this->_activation==ActivationType::NORM_CHAN_SOFTMAX_MAXVAL||
            this->_activation==ActivationType::NONE)
    {
        return;
    }

    if(_actParams.size() > 0)
    {
        Activations::activateArray(this->_output, this->_outputNum*this->_batch, this->_activation, this->supportAvx,_actParams[0]);
    }
    else
    {
        Activations::activateArray(this->_output, this->_outputNum*this->_batch, this->_activation, this->supportAvx);
    }

    auto so = std::chrono::system_clock::now();

    this->_forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

void ConnectedLayer::loadAllWeigths(std::vector<float> &weights)
{

    if(weights.size() != this->_numWeights)
    {
        throw Exception(1,"Connect weights load err. needed : " + std::to_string(this->_numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__);
    }

    loadWeights(weights.data(), _nWeights);

    if(this->_batchNorm)
    {
        loadScales(weights.data() + _nWeights, _nScales);
        loadRollMean(weights.data() + _nWeights + _nScales, _nRollMean);
        loadRollVariance(weights.data() + _nWeights + _nScales + _nRollMean, _nRollVariance);
        loadBias(weights.data() + _nWeights + _nScales + _nRollMean + _nRollVariance, _nBiases);
    }
    else
    {
        loadBias(weights.data() + _nWeights, _nBiases);
    }
}

void ConnectedLayer::loadScales(float * const &weights, const int &len)
{
    if(len != this->_nScales)
    {
        throw Exception(1, "load scales data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, weights, 1, this->_scales,1);
}

void ConnectedLayer::loadBias(float * const &bias, const int &len)
{
    if(len != this->_nBiases)
    {
        throw Exception(1, "load bias data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, bias, 1, this->_biases,1);
}

void ConnectedLayer::loadWeights(float * const &weights, const int &len)
{
    if(len != this->_nWeights)
    {
        throw Exception(1, "load weights data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, weights, 1, this->_weights,1);
}

void ConnectedLayer::loadRollMean(float * const &rollMean, const int &len)
{
    if(len != this->_nRollMean)
    {
        throw Exception(1, "load roll mean data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, rollMean, 1, this->_rollMean,1);
}

void ConnectedLayer::loadRollVariance(float * const &rollVariance, const int &len)
{
    if(len != this->_nRollVariance)
    {
        throw Exception(1, "load roll variance data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, rollVariance, 1, this->_rollVariance,1);
}

float *ConnectedLayer::getWeights() const
{
    return _weights;
}

float *ConnectedLayer::getBiases() const
{
    return _biases;
}

float *ConnectedLayer::getScales() const
{
    return _scales;
}

float *ConnectedLayer::getRollMean() const
{
    return _rollMean;
}

float *ConnectedLayer::getRollVariance() const
{
    return _rollVariance;
}

int ConnectedLayer::getNBiases() const
{
    return _nBiases;
}

int ConnectedLayer::getNWeights() const
{
    return _nWeights;
}

int ConnectedLayer::getNScales() const
{
    return _nScales;
}

int ConnectedLayer::getNRollMean() const
{
    return _nRollMean;
}

int ConnectedLayer::getNRollVariance() const
{
    return _nRollVariance;
}

int ConnectedLayer::getKSize() const
{
    return _kSize;
}

int ConnectedLayer::getStride() const
{
    return _stride;
}

int ConnectedLayer::getStrideX() const
{
    return _strideX;
}

int ConnectedLayer::getStrideY() const
{
    return _strideY;
}

int ConnectedLayer::getPadding() const
{
    return _padding;
}

int ConnectedLayer::getDilation() const
{
    return _dilation;
}

int ConnectedLayer::getBatchNorm() const
{
    return _batchNorm;
}
}
