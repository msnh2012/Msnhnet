#include "Msnhnet/layers/MsnhBatchNormLayer.h"
namespace Msnhnet
{
BatchNormLayer::BatchNormLayer(const int &batch, const int &width, const int &height, const int &channel, const ActivationType &activation, const std::vector<float> &actParams)
{
    this->type          =  LayerType::BATCHNORM;
    this->layerName     =  "BatchNorm       ";
    this->batch         =  batch;

    this->height        =  height;
    this->width         =  width;
    this->channel       =  channel;
    this->outHeight     =  height;
    this->outWidth      =  width;
    this->outChannel    =  channel;

    this->activation    =   activation;
    this->actParams     =   actParams;

    this->num           =  this->outChannel;

    this->inputNum      =  height * width * channel;

    this->outputNum     =  this->inputNum;
    this->nBiases       =  channel;
    this->nScales       =  channel;
    this->nRollMean     =  channel;
    this->nRollVariance =  channel;

    this->numWeights    =   static_cast<size_t>(this->nScales + this->nBiases + this->nRollMean + this->nRollVariance);

    if(!BaseLayer::isPreviewMode)
    {
        this->output        =  new float[static_cast<size_t>(this->outputNum * this->batch)]();

        this->biases        =  new float[static_cast<size_t>(channel)]();

        this->scales        =  new float[static_cast<size_t>(channel)]();

        this->rollMean      =  new float[static_cast<size_t>(channel)]();

        this->rollVariance  =  new float[static_cast<size_t>(channel)]();

        for(int i=0; i<channel; ++i)
        {
            this->scales[i] = 1;

        }
    }

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Batch Normalization Layer:     %d x %d x %d image\n", this->width, this->height, this->channel);
#else
    sprintf(msg, "Batch Normalization Layer:     %d x %d x %d image\n", this->width, this->height, this->channel);
#endif
    this->layerDetail   = msg;

}

void BatchNormLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    for (int b = 0; b < this->batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int c = 0; c < this->channel; ++c)
        {
#ifdef USE_ARM
            for (int i = 0; i < this->outHeight*this->outWidth; ++i)
            {
                int index = b*this->channel*this->outHeight*this->outWidth + c*this->outHeight*this->outWidth + i;

                this->output[index]  = this->scales[c]*(netState.input[index] - this->rollMean[c])/sqrt(this->rollVariance[c] + 0.00001f) + this->biases[c];
            }
#endif

#ifdef USE_X86
            if(this->supportAvx)
            {
                int i = 0;
                for (; i < (this->outHeight*this->outWidth)/8; ++i)
                {

                    int index = b*this->channel*this->outHeight*this->outWidth + c*this->outHeight*this->outWidth + i*8;

                    __m256 mScale;
                    __m256 mInput;
                    __m256 mMean;
                    __m256 mVariance;
                    __m256 mEsp;
                    __m256 mBias;
                    __m256 mResult1;
                    __m256 mResult2;

                    mScale      =   _mm256_set1_ps(this->scales[c]);
                    mInput      =   _mm256_loadu_ps(netState.input+index);
                    mMean       =   _mm256_set1_ps(this->rollMean[c]);
                    mVariance   =   _mm256_set1_ps(this->rollVariance[c]);
                    mEsp        =   _mm256_set1_ps(0.00001f);
                    mBias       =   _mm256_set1_ps(this->biases[c]);
                    mResult1    =   _mm256_sub_ps(mInput, mMean);
                    mResult1    =   _mm256_mul_ps(mScale, mResult1);
                    mResult2    =   _mm256_add_ps(mVariance,mEsp);
                    mResult2    =   _mm256_sqrt_ps(mResult2);

                    mResult2    =   _mm256_div_ps(mResult1,mResult2);
                    mResult2    =   _mm256_add_ps(mResult2,mBias);

                    _mm256_storeu_ps(this->output+index, mResult2);

                }

                for (int j = i*8; j < this->outHeight*this->outWidth; ++j)
                {
                    int index = b*this->channel*this->outHeight*this->outWidth + c*this->outHeight*this->outWidth + j;
                    this->output[index]  = this->scales[c]*(netState.input[index] - this->rollMean[c])/sqrt(this->rollVariance[c] + 0.00001f) + this->biases[c];
                }
            }
            else
            {
                for (int i = 0; i < this->outHeight*this->outWidth; ++i)
                {
                    int index = b*this->channel*this->outHeight*this->outWidth + c*this->outHeight*this->outWidth + i;
                    this->output[index]  = this->scales[c]*(netState.input[index] - this->rollMean[c])/sqrt(this->rollVariance[c] + 0.00001f) + this->biases[c];
                }
            }
#endif
        }

    }

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
            Activations::activateArray(this->output, this->outputNum*this->batch, this->activation, actParams[0]);
        }
        else
        {
            Activations::activateArray(this->output, this->outputNum*this->batch, this->activation);
        }
    }

    auto so = std::chrono::system_clock::now();
    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

void BatchNormLayer::addBias(float *const &output, float *const &biases, const int &batch, const int &channel, const int &whSize)
{
    for(int b=0; b<batch; ++b)

    {
        for(int i=0; i<channel; ++i)
        {
            for(int j=0; j<whSize; ++j)
            {
                output[(b*channel + i)*whSize + j] += biases[i];
            }
        }
    }
}

void BatchNormLayer::scaleBias(float *const &output, float *const &scales, const int &batch, const int &channel, const int &whSize)
{
    for(int b=0; b<batch; ++b)
    {
        for(int i=0; i<channel; ++i)

        {
            for(int j=0; j<whSize; ++j)

            {
                output[(b*channel + i)*whSize + j] *= scales[i];
            }
        }
    }
}

void BatchNormLayer::resize(int width, int height)
{
    this->outHeight = height;
    this->outWidth  = width;
    this->height    = height;
    this->width     = width;

    this->outputNum = height * width * this->channel;
    this->inputNum  = this->outputNum;

    const int outputSize = this->outputNum * this->batch;

    if(this->output == nullptr)
    {
        throw Exception(1,"output can't be null", __FILE__, __LINE__);
    }

    this->output    = static_cast<float*>(realloc(this->output, static_cast<size_t>(outputSize)*sizeof(float)));

}

void BatchNormLayer::loadAllWeigths(std::vector<float> &weights)
{

    if(weights.size() != this->numWeights)
    {
        throw Exception(1,"BatcnNorm weights load err. needed : " + std::to_string(this->numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__);
    }

    loadScales(weights.data(), nScales);
    loadBias(weights.data() + nScales , nBiases);
    loadRollMean(weights.data() + nScales + nBiases, nRollMean);
    loadRollVariance(weights.data() + nScales + nBiases + nRollVariance, nRollMean);
}

void BatchNormLayer::loadBias(float * const &bias, const int &len)
{
    if(len != this->nBiases)
    {
        throw Exception(1, "load bias data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, bias, 1, this->biases,1);
}

void BatchNormLayer::loadScales(float * const &weights, const int &len)
{
    if(len != this->nScales)
    {
        throw Exception(1, "load scales data len error",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, weights, 1, this->scales,1);
}

void BatchNormLayer::loadRollMean(float * const &rollMean, const int &len)
{
    if(len != this->channel)
    {
        throw Exception(1, "load roll mean data len error ",__FILE__,__LINE__);
    }

    Blas::cpuCopy(len, rollMean, 1, this->rollMean,1);
}

void BatchNormLayer::loadRollVariance(float * const &rollVariance, const int &len)
{
    if(len != this->channel)
    {
        throw Exception(1, "load roll variance data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, rollVariance, 1, this->rollVariance,1);
}

BatchNormLayer::~BatchNormLayer()
{
    releaseArr(scales);
    releaseArr(biases);
    releaseArr(rollMean);
    releaseArr(rollVariance);
    releaseArr(activationInput);
}
}
