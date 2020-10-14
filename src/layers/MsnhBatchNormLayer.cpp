#include "Msnhnet/layers/MsnhBatchNormLayer.h"
namespace Msnhnet
{
BatchNormLayer::BatchNormLayer(const int &batch, const int &width, const int &height, const int &channel, const ActivationType &activation, const float &eps, const std::vector<float> &actParams)
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
    this->_eps           =  eps;

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

    this->_maxOutputNum  = this->_batch*this->_outputNum;

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
    float* layerInput   = netState.getInput();
    float* layerOutput  = nullptr;

    /* 输入 */
    if(this->_isBranchLayer) 

    {
        if(this->_isFirstBranch)

        {
            layerInput      = netState.input;
        }
    }
    else
    {
        if(this->_layerIndex == 0) 

        {
            layerInput      = netState.input;
        }
        else 

        {
            if(netState.net->layers[this->_layerIndex - 1]->getMemReUse() == 0)

            {
                layerInput  = netState.input;
            }
        }
    }

    /* 输出 */
    if(this->_isBranchLayer) 

    {
        if(this->_isLastBranch)

        {
            layerOutput     = this->_output; 

        }
        else 

        {
            layerOutput     = netState.getOutput(); 

            netState.shuffleInOut();

        }
    }
    else
    {
        if(this->_memReUse==1) 

        {
            layerOutput     = netState.getOutput(); 

            netState.shuffleInOut();

        }
        else

        {
            layerOutput     = this->_output;
        }
    }

    for (int b = 0; b < this->_batch; ++b)
    {

#ifdef USE_ARM
#ifdef USE_NEON
        int step = b*this->_outChannel*this->_outHeight*this->_outWidth;
        BatchNormLayerArm::BatchNorm(layerInput + step,
                                     this->_width,
                                     this->_height,
                                     this->_channel,
                                     layerOutput + step,
                                     this->_scales,
                                     this->_rollMean,
                                     this->_rollVariance,
                                     this->_biases,
                                     this->_eps
                                     );
#else
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int c = 0; c < this->_outChannel; ++c)
        {
            float sqrtVal   = sqrt(this->_rollVariance[c] + this->_eps);
            float scaleSqrt = this->_scales[c]/sqrtVal;
            float meanSqrt  = -this->_scales[c]*this->_rollMean[c]/sqrtVal;

            for (int i = 0; i < this->_outHeight*this->_outWidth; ++i)
            {
                int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i;

                layerOutput[index]  = scaleSqrt*layerInput[index] + meanSqrt + this->_biases[c];
            }
        }
#endif
#endif

#ifdef USE_X86

        for (int c = 0; c < this->_outChannel; ++c)
        {
            float sqrtVal   = sqrt(this->_rollVariance[c] + this->_eps);
            float scaleSqrt = this->_scales[c]/sqrtVal;
            float meanSqrt  = -this->_scales[c]*this->_rollMean[c]/sqrtVal;

            if(this->supportAvx)
            {
                for (int i = 0; i < (this->_outHeight*this->_outWidth)/8; ++i)
                {

                    int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i*8;

                    __m256 mScaleSqrt;
                    __m256 mInput;
                    __m256 mMeanSqrt;
                    __m256 mBias;
                    __m256 mResult;

                    mScaleSqrt  =   _mm256_set1_ps(scaleSqrt);
                    mInput      =   _mm256_loadu_ps(layerInput+index);
                    mMeanSqrt   =   _mm256_set1_ps(meanSqrt);
                    mBias       =   _mm256_set1_ps(this->_biases[c]);
                    mResult     =   _mm256_mul_ps(mScaleSqrt, mInput);
                    mResult     =   _mm256_add_ps(mResult, mMeanSqrt);
                    mResult     =   _mm256_add_ps(mResult, mBias);

                    _mm256_storeu_ps(layerOutput+index, mResult);

                }

                for (int j = (this->_outHeight*this->_outWidth)/8*8; j < this->_outHeight*this->_outWidth; ++j)
                {
                    int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + j;
                    layerOutput[index]  = scaleSqrt*layerInput[index] + meanSqrt + this->_biases[c];
                }
            }
            else
            {

                for (int i = 0; i < this->_outHeight*this->_outWidth; ++i)
                {
                    int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i;
                    layerOutput[index]  = scaleSqrt*layerInput[index] + meanSqrt + this->_biases[c];
                }
            }
        }
#endif
    }

    if(this->_activation == ActivationType::NORM_CHAN)
    {
        Activations::activateArrayNormCh(layerOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                         this->_outWidth*this->_outHeight, layerOutput);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        Activations::activateArrayNormChSoftMax(layerOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, layerOutput,0);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        Activations::activateArrayNormChSoftMax(layerOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                this->_outWidth*this->_outHeight, layerOutput,1);
    }
    else if(this->_activation == ActivationType::NONE)
    {

    }
    else
    {                           

        if(_actParams.size() > 0)
        {
            Activations::activateArray(layerOutput, this->_outputNum*this->_batch, this->_activation, this->supportAvx, _actParams[0]);
        }
        else
        {
            Activations::activateArray(layerOutput, this->_outputNum*this->_batch, this->_activation, this->supportAvx);
        }
    }

    this->_forwardTime =   TimeUtil::getElapsedTime(st);

}

void BatchNormLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu) 

            {
                this->_output        =  new float[static_cast<size_t>(this->_outputNum * this->_batch)](); 

            }

#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuOutput        =   Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }

    this->_memReUse         =  0;
}

#ifdef USE_GPU
void BatchNormLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

    float* layerGpuInput   = netState.getGpuInput();
    float* layerGpuOutput  = nullptr;

    /* 输入 */
    if(this->_isBranchLayer) 

    {
        if(this->_isFirstBranch)

        {
            layerGpuInput      = netState.input;
        }
    }
    else
    {
        if(this->_layerIndex == 0) 

        {
            layerGpuInput      = netState.input;
        }
        else 

        {
            if(netState.net->layers[this->_layerIndex - 1]->getMemReUse() == 0)

            {
                layerGpuInput  = netState.input;
            }
        }
    }

    /* 输出 */
    if(this->_isBranchLayer) 

    {
        if(this->_isLastBranch)

        {
            layerGpuOutput     = this->_gpuOutput; 

        }
        else 

        {
            layerGpuOutput     = netState.getGpuOutput(); 

            netState.shuffleGpuInOut();

        }
    }
    else
    {
        if(this->_memReUse==1) 

        {
            layerGpuOutput     = netState.getGpuOutput(); 

            netState.shuffleGpuInOut();

        }
        else

        {
            layerGpuOutput     = this->_gpuOutput;
        }
    }

    BlasGPU::gpuSimpleCopy(this->_outputNum*this->_batch, layerGpuInput, layerGpuOutput);
    BlasGPU::gpuNorm(layerGpuOutput, this->_gpuRollMean, this->_gpuRollVariance, this->_batch, this->_outChannel, this->_eps, this->_outHeight *this->_outWidth);
    BlasGPU::gpuScaleBias(layerGpuOutput, this->_gpuScales, this->_batch, this->_outChannel, this->_outHeight*this->_outWidth);
    BlasGPU::gpuAddBias(layerGpuOutput, this->_gpuBiases, this->_batch, this->_outChannel, this->_outHeight*this->_outWidth);
    if(this->_activation == ActivationType::NORM_CHAN)
    {
        ActivationsGPU::gpuActivateArrayNormCh(layerGpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                               this->_outWidth*this->_outHeight, layerGpuOutput);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX)
    {
        ActivationsGPU::gpuActivateArrayNormChSoftMax(layerGpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                      this->_outWidth*this->_outHeight, layerGpuOutput,0);
    }
    else if(this->_activation == ActivationType::NORM_CHAN_SOFTMAX_MAXVAL)
    {
        ActivationsGPU::gpuActivateArrayNormChSoftMax(layerGpuOutput, this->_outputNum*this->_batch, this->_batch, this->_outChannel,
                                                      this->_outWidth*this->_outHeight, layerGpuOutput,1);
    }
    else if(this->_activation == ActivationType::NONE)
    {

    }
    else
    {                           

        if(_actParams.size() > 0)
        {
            ActivationsGPU::gpuActivateArray(layerGpuOutput, this->_outputNum*this->_batch, this->_activation, _actParams[0]);
        }
        else
        {
            ActivationsGPU::gpuActivateArray(layerGpuOutput, this->_outputNum*this->_batch, this->_activation);
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

    if(!BaseLayer::isPreviewMode)
    {
        this->_biases        =  new float[static_cast<size_t>(this->_nBiases)](); 

        this->_scales        =  new float[static_cast<size_t>(this->_nScales)](); 

        this->_rollMean      =  new float[static_cast<size_t>(this->_nRollMean)](); 

        this->_rollVariance  =  new float[static_cast<size_t>(this->_nRollVariance)](); 

        loadScales(weights.data(), this->_nScales);
        loadBias(weights.data() + this->_nScales , this->_nBiases);
        loadRollMean(weights.data() + this->_nScales + this->_nBiases, this->_nRollMean);
        loadRollVariance(weights.data() + this->_nScales + this->_nBiases + this->_nRollVariance, this->_nRollMean);

#ifdef USE_GPU
        if(!BaseLayer::onlyUseCpu)

        {
            this->_gpuBiases        = Cuda::makeCudaArray(this->_biases, this->_nBiases);
            this->_gpuScales        = Cuda::makeCudaArray(this->_scales, this->_nScales);
            this->_gpuRollMean      = Cuda::makeCudaArray(this->_rollMean, this->_nRollMean);
            this->_gpuRollVariance  = Cuda::makeCudaArray(this->_rollVariance, this->_nRollVariance);
        }

        if(BaseLayer::onlyUseGpu) 

        {
            releaseArr(this->_biases      );
            releaseArr(this->_scales      );
            releaseArr(this->_rollMean    );
            releaseArr(this->_rollVariance);
        }
#endif
    }

    this->_weightsLoaded = true;

}

void BatchNormLayer::saveWeights(const int &mainIdx, const int &branchIdx, const int &branchIdx1)
{

    if(BaseLayer::isPreviewMode)
    {
        throw Exception(1,"BatcnNorm preview mode can't save weights.", __FILE__, __LINE__, __FUNCTION__);
    }

    if(!this->_weightsLoaded)
    {
        throw Exception(1,"BatcnNorm weights had not been loaded yet.", __FILE__, __LINE__, __FUNCTION__);
    }

    std::string name = "";

    if(branchIdx!=-1)
    {
        name = "_" + std::to_string(mainIdx) + "_" + std::to_string(branchIdx) + "_" + std::to_string(branchIdx1) +".txt";
    }
    else
    {
        name = std::to_string(this->_layerIndex)+".txt";
    }

#ifdef USE_GPU
    if(BaseLayer::onlyUseGpu) 

    {
        Cuda::pullCudaArray(this->_gpuScales, this->_scales, this->_nScales);
        Cuda::pullCudaArray(this->_gpuBiases, this->_biases, this->_nBiases);
        Cuda::pullCudaArray(this->_gpuRollMean, this->_rollMean, this->_nRollMean);
        Cuda::pullCudaArray(this->_gpuRollVariance, this->_rollVariance, this->_nRollVariance);
    }
#endif

    std::vector<float> scalesVec(this->_scales,this->_scales+this->_nScales);
    std::vector<float> biasesVec(this->_biases,this->_biases+this->_nBiases);
    std::vector<float> rollMeanVec(this->_rollMean,this->_rollMean+this->_nRollMean);
    std::vector<float> rollVarianceVec(this->_rollVariance,this->_rollVariance+this->_nRollVariance);

    if(this->_scales==nullptr || this->_biases==nullptr || this->_rollMean==nullptr || this->_rollVariance==nullptr)
    {
        throw Exception(1,"BatcnNorm weights err.", __FILE__, __LINE__, __FUNCTION__);
    }

    std::string scaleName = "scale"+name;
    Msnhnet::IO::saveVector<float>(scalesVec,scaleName.c_str(),"\n");
    std::string biasName = "bias"+name;
    Msnhnet::IO::saveVector<float>(biasesVec,biasName.c_str(),"\n");
    std::string rollMeanName = "rollMean"+name;
    Msnhnet::IO::saveVector<float>(rollMeanVec,rollMeanName.c_str(),"\n");
    std::string rollVarianceName = "rollVariance"+name;
    Msnhnet::IO::saveVector<float>(rollVarianceVec,rollVarianceName.c_str(),"\n");
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
    releaseArr(_scales);
    releaseArr(_biases);
    releaseArr(_rollMean);
    releaseArr(_rollVariance);
    releaseArr(_activationInput);

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
