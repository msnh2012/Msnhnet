#include "Msnhnet/layers/MsnhConvolutionalLayer.h"

namespace Msnhnet
{
ConvolutionalLayer::ConvolutionalLayer(const int &batch, const int &steps, const int &height, const int &width, const int &channel, const int &num,
                                       const int &groups, const int &kSizeX, const int &kSizeY, const int &strideX, const int &strideY, const int &dilationX, const int &dilationY,
                                       const int &paddingX, const int &paddingY, ActivationType activation, const std::vector<float> &actParams, const int &batchNorm, const int &useBias, const int &binary, const int &xnor, const int &useBinOutput, const int &groupIndex, const int &antialiasing,
                                       ConvolutionalLayer * const &shareLayer, const int &assistedExcitation, const int &deform)
{

    (void) deform;
    int totalBatch          = batch * steps;
    this->type              = LayerType::CONVOLUTIONAL;

    if(xnor)
    {
        this->groups        = 1;

    }
    else
    {
        if(groups<1)
        {
            this->groups    = 1;
        }
        else
        {
            this->groups    = groups;
        }
    }

    this->antialiasing      = antialiasing;

    if(antialiasing)
    {
        this->stride        = 1;
        this->strideY       = 1;
        this->strideX       = 1;
    }
    else
    {
        this->stride        = strideX;
        this->strideX       = strideX;
        this->strideY       = strideY;
    }

    this->assistedExcite    = assistedExcitation;
    this->shareLayer        = shareLayer;
    this->groupIndex        = groupIndex;
    this->height            = height;
    this->width             = width;
    this->channel           = channel;
    this->num               = num;
    this->binary            = binary;
    this->xnor              = xnor;
    this->useBinOutput      = useBinOutput;
    this->batch             = batch;
    this->steps             = steps;
    this->dilationX         = dilationX;
    this->dilationY         = dilationY;
    this->kSizeX            = kSizeX;
    this->kSizeY            = kSizeY;
    this->paddingX          = paddingX;
    this->paddingY          = paddingY;
    this->batchNorm         = batchNorm;
    this->nWeights          = (this->channel / groups) * num * kSizeX * kSizeY;

    this->useBias           = useBias;

    if(this->useBias)
    {
        this->nBiases       = this->num;
    }
    else
    {
        this->nBiases       =   0;
    }

    if(this->shareLayer != nullptr)
    {
        if(     this->kSizeX    != this->shareLayer->kSizeX ||
                this->kSizeY    != this->shareLayer->kSizeY ||
                this->nWeights != this->shareLayer->nWeights||
                this->channel  != this->shareLayer->channel ||
                this->num      != this->shareLayer->num)
        {
            throw Exception(1, "Layer size, nweights, channels or filters don't match for the share_layer", __FILE__, __LINE__);
        }

        this->weights       = this->shareLayer->weights;
        this->biases        = this->shareLayer->biases;
    }
    else
    {
        if(!BaseLayer::isPreviewMode)
        {
            this->weights       = new float[static_cast<size_t>(this->nWeights)]();
            this->biases        = new float[static_cast<size_t>(this->num)]();
#ifdef USE_NNPACK
            this->nnBiases      = new float[static_cast<size_t>(this->num)]();
#endif
        }
    }

    this->outHeight         = convOutHeight();
    this->outWidth          = convOutWidth();
    this->outChannel        = num;

    this->outputNum         = this->outHeight * this->outWidth * this->outChannel;
    this->inputNum          = height * width * channel;

    this->activation        = activation;
    this->actParams         = actParams;

    this->output            = new float[static_cast<size_t>(outputNum * this->batch)]();

    if(binary)
    {
        if(!BaseLayer::isPreviewMode)
        {
            this->binaryWeights = new float[static_cast<size_t>(this->nWeights)]();
            this->cWeights      = new char[static_cast<size_t>(this->nWeights)]();
            this->scales        = new float[static_cast<size_t>(this->num)]();
        }
    }

    if(xnor)
    {

        int align           = 32;

        int srcAlign        = this->outHeight * this->outWidth;
        this->bitAlign      = srcAlign + (align - srcAlign % align);
        this->ldaAlign      = 256;

        if(!BaseLayer::isPreviewMode)
        {
            this->binaryWeights = new float[static_cast<size_t>(this->nWeights)]();
            this->binaryInputs  = new float[static_cast<size_t>(this->inputNum * this->batch)]();
            this->meanArr       = new float[static_cast<size_t>(this->num)]();

            const int newCh     = this->channel / 32;
            int rePackedISize   = newCh * this->width * this->height + 1;
            this->binRePackedIn = new uint32_t[static_cast<size_t>(rePackedISize)]();

            int k               = this->kSizeX * this->kSizeY * this->channel;
            int kAligned        = k + (this->ldaAlign - k%this->ldaAlign);
            int tBitInSize      = kAligned * this->bitAlign / 8;
            this->tBitInput     = new char[static_cast<size_t>(tBitInSize)]();
        }
    }

    if(batchNorm)
    {
        if(this->shareLayer!=nullptr)
        {
            this->scales            = this->shareLayer->scales;
            this->rollMean          = this->shareLayer->rollMean;
            this->rollVariance      = this->shareLayer->rollVariance;
        }
        else
        {
            if(!BaseLayer::isPreviewMode)
            {
                this->scales         = new float[static_cast<size_t>(this->num)]();
                this->rollMean       = new float[static_cast<size_t>(this->num)]();
                this->rollVariance   = new float[static_cast<size_t>(this->num)]();
            }
        }

        this->nScales           =   num;
        this->nRollMean         =   num;
        this->nRollVariance     =   num;
    }

    this->numWeights            =   static_cast<size_t>(this->nWeights + this->nScales + this->nRollMean + this->nRollVariance + this->nBiases);

#ifndef GPU
    if(this->activation == ActivationType::SWISH || this->activation == ActivationType::MISH)
    {
        this->activationInput = new float[static_cast<size_t>(this->outputNum * totalBatch)]();
    }
#endif

    this->workSpaceSize = getConvWorkSpaceSize();

    this->bFlops        = (2.0f * this->nWeights * this->outHeight * this->outWidth) / 1000000000.f;

    if(this->xnor)
    {
        this->bFlops    = this->bFlops / 32;
    }

    if(this->xnor && this->useBinOutput)
    {
        this->layerName = "ConvXB          ";
        this->layerDetail.append("convXB");
    }
    else if(this->xnor)
    {
        this->layerName = "ConvX           ";
        this->layerDetail.append("convX ");
    }
    else if(this->shareLayer != nullptr)
    {
        this->layerName = "ConvS           ";
        this->layerDetail.append("convS ");
    }
    else if(this->assistedExcite)
    {
        this->layerName = "ConvAE          ";
        this->layerDetail.append("convAE");
    }
    else if(this->batchNorm)
    {
        this->layerName = "ConvBN          ";
        this->layerDetail.append("convBN");
    }
    else if(this->groups == this->num)
    {
        this->layerName = "ConvDW          ";
        this->layerDetail.append("convDW");
    }
    else
    {
        this->layerName = "Conv            ";
        this->layerDetail.append("conv  ");
    }

    char str[100];
    if(this->groups > 1)
    {
#ifdef WIN32
        sprintf_s(str,"%5d/%4d ", this->num, this->groups);
#else
        sprintf(str,"%5d/%4d ", this->num, this->groups);
#endif
    }
    else
    {
#ifdef WIN32
        sprintf_s(str,"%5d      ", this->num);
#else
        sprintf(str,"%5d      ", this->num);
#endif
    }

    this->layerDetail.append(std::string(str));

    if(this->strideX != this->strideY)
    {
#ifdef WIN32
        sprintf_s(str,"%2dx%2d/%2dx%2d ", this->kSizeX, this->kSizeY, this->strideX, this->strideY);
#else
        sprintf(str,"%2dx%2d/%2dx%2d ", this->kSizeX, this->kSizeY, this->strideX, this->strideY);
#endif
    }
    else
    {
        if(this->dilationX > 1)
        {
#ifdef WIN32
            sprintf_s(str,"%2d x%2d/%2d(%1d)", this->kSizeX, this->kSizeY, this->strideX, this->dilationX);
#else
            sprintf(str,"%2d x%2d/%2d(%1d)", this->kSizeX, this->kSizeY, this->strideX, this->dilationX);
#endif
        }
        else
        {
#ifdef WIN32
            sprintf_s(str, "%2d x%2d/%2d   ", this->kSizeX, this->kSizeY, this->strideX);
#else
            sprintf(str, "%2d x%2d/%2d   ", this->kSizeX, this->kSizeY, this->strideX);
#endif
        }
    }

    this->layerDetail.append(std::string(str));

#ifdef WIN32
    sprintf_s(str, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->width, this->height, this->channel,
              this->outWidth, this->outHeight, this->outChannel, static_cast<double>(this->bFlops));
#else
    sprintf(str, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->width, this->height, this->channel,
            this->outWidth, this->outHeight, this->outChannel, static_cast<double>(this->bFlops));
#endif

    this->layerDetail.append(std::string(str));

}

ConvolutionalLayer::~ConvolutionalLayer()
{
    releaseArr(weights);
    releaseArr(biases);
    releaseArr(scales);
    releaseArr(rollMean);
    releaseArr(rollVariance);
    releaseArr(cWeights);
    releaseArr(binaryInputs);
    releaseArr(binaryWeights);
    releaseArr(activationInput);
    releaseArr(meanArr);
    releaseArr(binRePackedIn);
    releaseArr(tBitInput);
    releaseArr(alignBitWeights);
}

int ConvolutionalLayer::convOutHeight()
{
    return (this->height + 2*this->paddingY - this->kSizeY)/this->strideY + 1;
}

int ConvolutionalLayer::convOutWidth()
{
    return (this->width + 2*this->paddingX - this->kSizeX)/this->strideX + 1;
}

int ConvolutionalLayer::getWorkSpaceSize32()
{
    if(this->xnor)
    {
        int  rePackInSize   = this->channel * this->width * this->height * static_cast<int>(sizeof(float));
        int  workSpaceSize  = this->bitAlign * this->kSizeX * this->kSizeY * this->channel * static_cast<int>(sizeof(float));
        if(workSpaceSize < rePackInSize)
        {
            workSpaceSize = rePackInSize;
            return workSpaceSize;
        }
    }

    return this->outHeight * this->outWidth * this->kSizeX * this->kSizeY * (this->channel / this->groups)*static_cast<int>(sizeof(float));
}

int ConvolutionalLayer::getWorkSpaceSize16()
{
    return 0;
}

void ConvolutionalLayer::addBias(float *const &output, float *const &biases, const int &batch, const int &num, const int &whSize)
{
    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < num; ++i)
        {
            for (int j = 0; j < whSize; ++j)
            {
                output[(b*num + i)*whSize + j] += biases[i];
            }
        }
    }
}

void ConvolutionalLayer::scaleBias(float *const &output, float *const &scales, const int &batch, const int &num, const int &whSize)
{
    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < num; ++i)
        {
            for (int j = 0; j < whSize; ++j)
            {
                output[(b*num + i)*whSize + j] *= scales[i];
            }
        }
    }
}

int ConvolutionalLayer::getConvWorkSpaceSize()
{
    int workSpaceSize32 = getWorkSpaceSize32();
    int workSpaceSize16 = getWorkSpaceSize16();

    if(workSpaceSize16 > workSpaceSize32)
    {
        workSpaceSize32 = workSpaceSize16;
    }

    return workSpaceSize32;
}

void ConvolutionalLayer::binarizeWeights(float * const &weights, const int &num, const int &wtSize, float * const &binary)
{
    for (int f = 0; f < num; ++f)
    {
        float mean  = 0;

        for (int i = 0; i < wtSize; ++i)
        {
            mean += fabs(weights[f*wtSize + i]);
        }

        mean    = mean / wtSize;

        for (int i = 0; i < wtSize; ++i)
        {
            binary[f*wtSize + i] = (weights[f*wtSize + i] > 0) ? mean :-mean;
        }
    }
}

void ConvolutionalLayer::cpuBinarize(float * const &x, const int &xNum, float * const &binary)
{
    for (int i = 0; i < xNum; ++i)
    {
        binary[i] = (x[i] > 0)? 1.f: -1.f;
    }
}

void ConvolutionalLayer::swapBinary()
{
    float *swapV         = this->weights;
    this->weights       = this->binaryWeights;
    this->binaryWeights = swapV;
}

void ConvolutionalLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    int mOutHeight      = convOutHeight();
    int mOutWidth       = convOutWidth();

#ifdef USE_NNPACK
    struct nnp_size     nnInSize    = {static_cast<size_t>(this->width),static_cast<size_t>(this->height)};
    struct nnp_padding  nnInPadding = {static_cast<size_t>(this->paddingX),static_cast<size_t>(this->paddingX),
                                      static_cast<size_t>(this->paddingY),static_cast<size_t>(this->paddingY)
                                      };
    struct nnp_size     nnKSize     = {static_cast<size_t>(this->kSizeX),static_cast<size_t>(this->kSizeY)};
    struct nnp_size     nnStride    = {static_cast<size_t>(this->strideX),static_cast<size_t>(this->strideY)};
#endif

    Blas::cpuFill(this->outputNum * this->batch, 0, this->output, 1);

    if(this->xnor && (!this->alignBitWeights))
    {
        if(!this->alignBitWeights)
        {
            binarizeWeights(this->weights,this->num, this->nWeights,this->binaryWeights);
        }
        swapBinary();

        cpuBinarize(netState.input, this->channel * this->height * this->width * this->batch, this->binaryInputs);
        netState.input = this->binaryInputs;
    }

    int m       =  this->num / this->groups;

    int k       =  this->kSizeX * this->kSizeY *this->channel / this->groups;

    int n       =  mOutHeight * mOutWidth;

    for (int i = 0; i < this->batch; ++i)
    {

        for (int j = 0; j < this->groups; ++j)
        {

            float *a    =  this->weights + j*this->nWeights /this->groups;

            float *b    =  netState.workspace;

            float *c    =  this->output + (i*this->groups +j)*n*m;

            if(this->xnor && this->alignBitWeights && this->strideX == this->strideY)
            {
                /* TODO: */

            }
            else
            {

                float *im = netState.input + (i*this->groups + j)*(this->channel / this->groups)*this->height*this->width;

#ifdef USE_NNPACK
                nnp_status status;
                status = nnp_convolution_inference(nnp_convolution_algorithm_implicit_gemm,
                                          nnp_convolution_transform_strategy_tuple_based,
                                          static_cast<size_t>(this->channel/this->groups),
                                          static_cast<size_t>(m),
                                          nnInSize,
                                          nnInPadding,
                                          nnKSize,
                                          nnStride,
                                          im,
                                          a,
                                          nullptr,
                                          c,
                                          nullptr,
                                          nullptr,
                                          nnp_activation_identity,
                                          nullptr,
                                          nullptr,
                                          nullptr
                                          );
                if(status !=0 )
                {
                    throw Exception(1,"NNPack error, code : "+std::to_string(status),__FILE__,__LINE__);
                }
#else
                if(this->kSizeX == 1 && this->kSizeY == 1 &&  this->strideX == 1  &&  this->strideY == 1&& this->paddingX == 0 && this->paddingY == 0)
                {
                    b = im;

                }
                else
                {

                    Gemm::cpuIm2colEx(im, this->channel/this->groups, this->height, this->width, this->kSizeX, this->kSizeY,
                                      this->paddingX, this->paddingY, this->strideX, this->strideY, this->dilationX, this->dilationY,
                                      b);

                }

                Gemm::cpuGemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n, this->supportAvx&&this->supportFma);
#endif
            }

        }

    }

    if(this->batchNorm==1)
    {

        for (int b = 0; b < this->batch; ++b)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int c = 0; c < this->outChannel; ++c)
            {
#ifdef USE_ARM
                for (int i = 0; i < this->outHeight*this->outWidth; ++i)
                {
                    int index = b*this->outChannel*this->outHeight*this->outWidth + c*this->outHeight*this->outWidth + i;

                    this->output[index]  = this->scales[c]*(this->output[index] - this->rollMean[c])/sqrt(this->rollVariance[c] + 0.00001f) + this->biases[c];
                }
#endif

#ifdef USE_X86
                if(this->supportAvx)
                {
                    int i = 0;
                    for (; i < (this->outHeight*this->outWidth)/8; ++i)
                    {

                        int index = b*this->outChannel*this->outHeight*this->outWidth + c*this->outHeight*this->outWidth + i*8;

                        __m256 mScale;
                        __m256 mInput;
                        __m256 mMean;
                        __m256 mVariance;
                        __m256 mEsp;
                        __m256 mBias;
                        __m256 mResult1;
                        __m256 mResult2;

                        mScale      =   _mm256_set1_ps(this->scales[c]);
                        mInput      =   _mm256_loadu_ps(this->output+index);
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
                        int index = b*this->outChannel*this->outHeight*this->outWidth + c*this->outHeight*this->outWidth + j;
                        this->output[index]  = this->scales[c]*(this->output[index] - this->rollMean[c])/sqrt(this->rollVariance[c] + 0.00001f) + this->biases[c];
                    }
                }
                else
                {
                    for (int i = 0; i < this->outHeight*this->outWidth; ++i)
                    {
                        int index = b*this->outChannel*this->outHeight*this->outWidth + c*this->outHeight*this->outWidth + i;

                        this->output[index]  = this->scales[c]*(this->output[index] - this->rollMean[c])/sqrt(this->rollVariance[c] + 0.00001f) + this->biases[c];
                    }
                }
#endif
            }
        }

    }
    else
    {
        if(useBias == 1)
            addBias(this->output, this->biases, this->batch, this->num, mOutHeight*mOutWidth);
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

    if(this->binary || this->xnor)
    {
        swapBinary();
    }

    auto so = std::chrono::system_clock::now();

    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

void ConvolutionalLayer::loadAllWeigths(std::vector<float> &weights)
{
    if(weights.size() != this->numWeights)
    {
        throw Exception(1,"Conv weights load err. needed : " + std::to_string(this->numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__);
    }

    loadWeights(weights.data(), nWeights);

    if(this->batchNorm)
    {
        loadScales(weights.data() + nWeights, nScales);
        loadBias(weights.data() + nWeights + nScales, nBiases);
        loadRollMean(weights.data() + nWeights + nScales + nBiases, nRollMean);
        loadRollVariance(weights.data() + nWeights + nScales + nBiases + nRollMean, nRollVariance);
    }
    else
    {
        if(useBias==1)
        {
            loadBias(weights.data() + nWeights, nBiases);
        }
    }
}

void ConvolutionalLayer::loadScales(float * const &weights, const int &len)
{
    if(len != this->nScales)
    {
        throw Exception(1, "load scales data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, weights, 1, this->scales,1);
}

void ConvolutionalLayer::loadBias(float * const &bias, const int &len)
{
    if(len != this->nBiases)
    {
        throw Exception(1, "load bias data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, bias, 1, this->biases,1);
}

void ConvolutionalLayer::loadWeights(float * const &weights, const int &len)
{
    if(len != this->nWeights)
    {
        throw Exception(1, "load weights data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, weights, 1, this->weights,1);
}

void ConvolutionalLayer::loadRollMean(float * const &rollMean, const int &len)
{
    if(len != this->nRollMean)
    {
        throw Exception(1, "load roll mean data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, rollMean, 1, this->rollMean,1);
}

void ConvolutionalLayer::loadRollVariance(float * const &rollVariance, const int &len)
{
    if(len != this->nRollVariance)
    {
        throw Exception(1, "load roll variance data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, rollVariance, 1, this->rollVariance,1);
}
}
