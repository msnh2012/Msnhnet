#include "Msnhnet/layers/MsnhBatchNormLayer.h"
namespace Msnhnet
{
BatchNormLayer::BatchNormLayer(const int &batch, const int &width, const int &height, const int &channel, const ActivationType &activation, const std::vector<float> &actParams)
{
    this->_type          =  LayerType::BATCHNORM;
    this->_layerName     =  "BatchNorm       ";
    this->_batch         =  batch;

    this->_height        =  height;
    this->_width         =  width;
    this->_channel       =  channel;
    this->_outHeight     =  height;
    this->_outWidth      =  width;
    this->_outChannel    =  channel;

    this->_activation    =   activation;
    this->_actParams     =   actParams;

    this->_num           =  this->_outChannel;

    this->_inputNum      =  height * width * channel;

    this->_outputNum     =  this->_inputNum;
    this->_nBiases       =  channel;
    this->_nScales       =  channel;
    this->_nRollMean     =  channel;
    this->_nRollVariance =  channel;

    this->_numWeights    =   static_cast<size_t>(this->_nScales + this->_nBiases + this->_nRollMean + this->_nRollVariance);

    if(!BaseLayer::isPreviewMode)
    {
        this->_output        =  new float[static_cast<size_t>(this->_outputNum * this->_batch)](); 

        this->_biases        =  new float[static_cast<size_t>(channel)](); 

        this->_scales        =  new float[static_cast<size_t>(channel)](); 

        this->_rollMean      =  new float[static_cast<size_t>(channel)](); 

        this->_rollVariance  =  new float[static_cast<size_t>(channel)](); 

        for(int i=0; i<channel; ++i)
        {
            this->_scales[i] = 1;                   

        }

#ifdef USE_GPU
        this->_gpuOutput     = Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
        this->_gpuBiases     = Cuda::makeCudaArray(this->_biases, this->_channel);
        this->_gpuScales     = Cuda::makeCudaArray(this->_scales, this->_channel);
        this->_gpuRollMean      = Cuda::makeCudaArray(this->_rollMean, this->_channel);
        this->_gpuRollVariance  = Cuda::makeCudaArray(this->_rollVariance, this->_channel);
#endif
    }

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Batch Normalization Layer:     %d x %d x %d image\n", this->_width, this->_height, this->_channel);
#else
    sprintf(msg, "Batch Normalization Layer:     %d x %d x %d image\n", this->_width, this->_height, this->_channel);
#endif
    this->_layerDetail   = msg;

}

void BatchNormLayer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();
    for (int b = 0; b < this->_batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int c = 0; c < this->_channel; ++c)
        {
#ifdef USE_ARM
            for (int i = 0; i < this->_outHeight*this->_outWidth; ++i)
            {
                int index = b*this->_channel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i;

                this->_output[index]  = this->_scales[c]*(netState.input[index] - this->_rollMean[c])/sqrt(this->_rollVariance[c] + 0.00001f) + this->_biases[c];
            }
#endif

#ifdef USE_X86
            if(this->supportAvx)
            {
                int i = 0;
                for (; i < (this->_outHeight*this->_outWidth)/8; ++i)
                {

                    int index = b*this->_channel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i*8;

                    __m256 mScale;
                    __m256 mInput;
                    __m256 mMean;
                    __m256 mVariance;
                    __m256 mEsp;
                    __m256 mBias;
                    __m256 mResult1;
                    __m256 mResult2;

                    mScale      =   _mm256_set1_ps(this->_scales[c]);
                    mInput      =   _mm256_loadu_ps(netState.input+index);
                    mMean       =   _mm256_set1_ps(this->_rollMean[c]);
                    mVariance   =   _mm256_set1_ps(this->_rollVariance[c]);
                    mEsp        =   _mm256_set1_ps(0.00001f);
                    mBias       =   _mm256_set1_ps(this->_biases[c]);
                    mResult1    =   _mm256_sub_ps(mInput, mMean);
                    mResult1    =   _mm256_mul_ps(mScale, mResult1);
                    mResult2    =   _mm256_add_ps(mVariance,mEsp);
                    mResult2    =   _mm256_sqrt_ps(mResult2);

                    mResult2    =   _mm256_div_ps(mResult1,mResult2);
                    mResult2    =   _mm256_add_ps(mResult2,mBias);

                    _mm256_storeu_ps(this->_output+index, mResult2);

                }

                for (int j = i*8; j < this->_outHeight*this->_outWidth; ++j)
                {
                    int index = b*this->_channel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + j;
                    this->_output[index]  = this->_scales[c]*(netState.input[index] - this->_rollMean[c])/sqrt(this->_rollVariance[c] + 0.00001f) + this->_biases[c];
                }
            }
            else
            {
                for (int i = 0; i < this->_outHeight*this->_outWidth; ++i)
                {
                    int index = b*this->_channel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i;
                    this->_output[index]  = this->_scales[c]*(netState.input[index] - this->_rollMean[c])/sqrt(this->_rollVariance[c] + 0.00001f) + this->_biases[c];
                }
            }
#endif
        }

    }

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

    this->_forwardTime =   TimeUtil::getElapsedTime(st);

}

#ifdef USE_GPU
void BatchNormLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

    BlasGPU::gpuSimpleCopy(this->_outputNum*this->_batch, netState.input, this->_gpuOutput);
    BlasGPU::gpuNorm(this->_gpuOutput, this->_gpuRollMean, this->_gpuRollVariance, this->_batch, this->_outChannel, this->_outHeight *this->_outWidth);
    BlasGPU::gpuScaleBias(this->_gpuOutput, this->_gpuScales, this->_batch, this->_outChannel, this->_outHeight*this->_outWidth);
    BlasGPU::gpuAddBias(this->_gpuOutput, this->_gpuBiases, this->_batch, this->_outChannel, this->_outHeight*this->_outWidth);
    if(this->_activation == ActivationType::NORM_CHAN)
    {
        ActivationsGPU::gpuActivateArrayNormCh(this->_gpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                            this->_outWidth*this->_outHeight, this->_gpuOutput);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        ActivationsGPU::gpuActivateArrayNormChSoftMax(this->_gpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, this->_gpuOutput,0);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        ActivationsGPU::gpuActivateArrayNormChSoftMax(this->_gpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, this->_gpuOutput,1);
    }
    else if(this->_activation == ActivationType::NONE)
    {

    }
    else
    {                           

        if(_actParams.size() > 0)
        {
            ActivationsGPU::gpuActivateArray(this->_gpuOutput, this->_outputNum*this->_batch, this->_activation, _actParams[0]);
        }
        else
        {
            ActivationsGPU::gpuActivateArray(this->_gpuOutput, this->_outputNum*this->_batch, this->_activation);
        }
    }

    this->recordCudaStop();
}
#endif

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
    this->_outHeight = height;
    this->_outWidth  = width;
    this->_height    = height;
    this->_width     = width;

    this->_outputNum = height * width * this->_channel;
    this->_inputNum  = this->_outputNum;

    const int outputSize = this->_outputNum * this->_batch;

    if(this->_output == nullptr)
    {
        throw Exception(1,"output can't be null", __FILE__, __LINE__, __FUNCTION__);
    }

    this->_output    = static_cast<float*>(realloc(this->_output, static_cast<size_t>(outputSize)*sizeof(float)));

}

void BatchNormLayer::loadAllWeigths(std::vector<float> &weights)
{

    if(weights.size() != this->_numWeights)
    {
        throw Exception(1,"BatcnNorm weights load err. needed : " + std::to_string(this->_numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__, __FUNCTION__);
    }

    loadScales(weights.data(), _nScales);
    loadBias(weights.data() + _nScales , _nBiases);
    loadRollMean(weights.data() + _nScales + _nBiases, _nRollMean);
    loadRollVariance(weights.data() + _nScales + _nBiases + _nRollVariance, _nRollMean);

#ifdef USE_GPU
    Cuda::pushCudaArray(this->_gpuScales, this->_scales, this->_nScales);
    Cuda::pushCudaArray(this->_gpuBiases, this->_biases, this->_nBiases);
    Cuda::pushCudaArray(this->_gpuRollMean, this->_rollMean, this->_nRollMean);
    Cuda::pushCudaArray(this->_gpuRollVariance, this->_rollVariance, this->_nRollVariance);
#endif
}

void BatchNormLayer::loadBias(float * const &bias, const int &len)
{
    if(len != this->_nBiases)
    {
        throw Exception(1, "load bias data len error ",__FILE__,__LINE__, __FUNCTION__);
    }
    Blas::cpuCopy(len, bias, 1, this->_biases,1);
}

void BatchNormLayer::loadScales(float * const &weights, const int &len)
{
    if(len != this->_nScales)
    {
        throw Exception(1, "load scales data len error",__FILE__,__LINE__, __FUNCTION__);
    }
    Blas::cpuCopy(len, weights, 1, this->_scales,1);
}

void BatchNormLayer::loadRollMean(float * const &rollMean, const int &len)
{
    if(len != this->_channel)
    {
        throw Exception(1, "load roll mean data len error ",__FILE__,__LINE__, __FUNCTION__);
    }

    Blas::cpuCopy(len, rollMean, 1, this->_rollMean,1);
}

void BatchNormLayer::loadRollVariance(float * const &rollVariance, const int &len)
{
    if(len != this->_channel)
    {
        throw Exception(1, "load roll variance data len error ",__FILE__,__LINE__, __FUNCTION__);
    }
    Blas::cpuCopy(len, rollVariance, 1, this->_rollVariance,1);
}

BatchNormLayer::~BatchNormLayer()
{
    releaseAll(_scales);
    releaseAll(_biases);
    releaseAll(_rollMean);
    releaseAll(_rollVariance);
    releaseAll(_activationInput);

#ifdef USE_GPU
    Cuda::freeCuda(_gpuScales);
    Cuda::freeCuda(_gpuBiases);
    Cuda::freeCuda(_gpuRollMean);
    Cuda::freeCuda(_gpuRollVariance);
#endif
}

float *BatchNormLayer::getScales() const
{
    return _scales;
}

float *BatchNormLayer::getBiases() const
{
    return _biases;
}

float *BatchNormLayer::getRollMean() const
{
    return _rollMean;
}

float *BatchNormLayer::getRollVariance() const
{
    return _rollVariance;
}

float *BatchNormLayer::getActivationInput() const
{
    return _activationInput;
}

int BatchNormLayer::getNBiases() const
{
    return _nBiases;
}

int BatchNormLayer::getNScales() const
{
    return _nScales;
}

int BatchNormLayer::getNRollMean() const
{
    return _nRollMean;
}

int BatchNormLayer::getNRollVariance() const
{
    return _nRollVariance;
}
}
