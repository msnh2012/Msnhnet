#include "Msnhnet/layers/MsnhConnectedLayer.h"
namespace Msnhnet
{
ConnectedLayer::ConnectedLayer(const int &batch, const int &steps, const int &inputNum,
                               const int &outputNum, const ActivationType &activation, const std::vector<float> &actParams, const int &batchNorm)
{
    int totalBatch      =   batch*steps;
    this->type          =   LayerType::CONNECTED;
    this->layerName     =  "Connected       ";

    this->inputNum      =   inputNum;
    this->outputNum     =   outputNum;
    this->batch         =   batch;
    this->batchNorm     =   batchNorm;

    this->height        =   1;
    this->width         =   1;
    this->channel       =   inputNum;

    this->outHeight     =   1;
    this->outWidth      =   1;
    this->outChannel    =   outputNum;

    this->num           =   this->outChannel;
    this->kSize         =   1;
    this->stride        =   1;
    this->padding       =   0;

    this->activation    =   activation;
    this->actParams     =   actParams;

    this->nWeights      =   inputNum * outputNum;
    this->nBiases       =   outputNum;

    if(!BaseLayer::isPreviewMode)
    {
        this->output        =   new float[static_cast<size_t>(totalBatch * outputNum) ]();
        this->weights       =   new float[static_cast<size_t>(inputNum * outputNum)]();
        this->biases        =   new float[static_cast<size_t>(outputNum)]();
    }

    if(batchNorm)
    {
        if(!BaseLayer::isPreviewMode)
        {
            this->scales        =   new float[static_cast<size_t>(outputNum)]();

            for (int i = 0; i < outputNum; ++i)
            {
                this->scales[i] =   1;
            }

            this->rollMean      =   new float[static_cast<size_t>(outputNum)]();
            this->rollVariance  =   new float[static_cast<size_t>(outputNum)]();
        }

        this->nScales       =   outputNum;
        this->nRollMean     =   outputNum;
        this->nRollVariance =   outputNum;
    }

    this->numWeights            =   static_cast<size_t>(this->nWeights + this->nScales + this->nRollMean + this->nRollVariance + this->nBiases);

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "connected                            %4d  ->  %4d\n", inputNum, outputNum);
#else
    sprintf(msg, "connected                            %4d  ->  %4d\n", inputNum, outputNum);
#endif
    this->layerDetail       = msg;
}

ConnectedLayer::~ConnectedLayer()
{
    releaseArr(weights);
    releaseArr(biases);
    releaseArr(scales);
    releaseArr(rollMean);
    releaseArr(rollVariance);

}

void ConnectedLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    Blas::cpuFill(this->outputNum * this->batch, 0, this->output, 1);
    int m       =   this->batch;
    int k       =   this->inputNum;
    int n       =   this->outputNum;

    float *a    =   netState.input;
    float *b    =   this->weights;
    float *c    =   this->output;

    Gemm::cpuGemm(0,1,m,n,k,1,a,k,b,k,1,c,n,this->supportAvx&&this->supportFma);

    if(this->batchNorm == 1)
    {

        Blas::cpuNorm(this->output, this->rollMean, this->rollVariance, this->batch, this->outputNum, 1);

        ConvolutionalLayer::scaleBias(this->output, this->scales, this->batch, this->outputNum, 1);

    }

    for (int i = 0; i < this->batch; ++i)
    {
        Blas::cpuAxpy(this->outputNum, 1, this->biases, 1, this->output + i * this->outputNum, 1);
    }

    if(     this->activation==ActivationType::NORM_CHAN||
            this->activation==ActivationType::NORM_CHAN_SOFTMAX||
            this->activation==ActivationType::NORM_CHAN_SOFTMAX_MAXVAL||
            this->activation==ActivationType::NONE)
    {
        return;
    }

    if(actParams.size() > 0)
    {
        Activations::activateArray(this->output, this->outputNum*this->batch, this->activation, this->supportAvx,actParams[0]);
    }
    else
    {
        Activations::activateArray(this->output, this->outputNum*this->batch, this->activation, this->supportAvx);
    }

    auto so = std::chrono::system_clock::now();

    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

void ConnectedLayer::loadAllWeigths(std::vector<float> &weights)
{

    if(weights.size() != this->numWeights)
    {
        throw Exception(1,"Connect weights load err. needed : " + std::to_string(this->numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__);
    }

    loadWeights(weights.data(), nWeights);

    if(this->batchNorm)
    {
        loadScales(weights.data() + nWeights, nScales);
        loadRollMean(weights.data() + nWeights + nScales, nRollMean);
        loadRollVariance(weights.data() + nWeights + nScales + nRollMean, nRollVariance);
        loadBias(weights.data() + nWeights + nScales + nRollMean + nRollVariance, nBiases);
    }
    else
    {
        loadBias(weights.data() + nWeights, nBiases);
    }
}

void ConnectedLayer::loadScales(float * const &weights, const int &len)
{
    if(len != this->nScales)
    {
        throw Exception(1, "load scales data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, weights, 1, this->scales,1);
}

void ConnectedLayer::loadBias(float * const &bias, const int &len)
{
    if(len != this->nBiases)
    {
        throw Exception(1, "load bias data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, bias, 1, this->biases,1);
}

void ConnectedLayer::loadWeights(float * const &weights, const int &len)
{
    if(len != this->nWeights)
    {
        throw Exception(1, "load weights data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, weights, 1, this->weights,1);
}

void ConnectedLayer::loadRollMean(float * const &rollMean, const int &len)
{
    if(len != this->nRollMean)
    {
        throw Exception(1, "load roll mean data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, rollMean, 1, this->rollMean,1);
}

void ConnectedLayer::loadRollVariance(float * const &rollVariance, const int &len)
{
    if(len != this->nRollVariance)
    {
        throw Exception(1, "load roll variance data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, rollVariance, 1, this->rollVariance,1);
}
}
