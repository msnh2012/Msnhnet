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
    this->_type              = LayerType::CONVOLUTIONAL;
    if(xnor)
    {
        this->_groups        = 1; 

    }
    else
    {
        if(groups<1)
        {
            this->_groups    = 1;
        }
        else
        {
            this->_groups    = groups;
        }
    }

    this->_antialiasing      = antialiasing;

    if(antialiasing)
    {
        this->_stride        = 1;
        this->_strideY       = 1;
        this->_strideX       = 1;
    }
    else
    {
        this->_stride        = strideX;
        this->_strideX       = strideX;
        this->_strideY       = strideY;
    }

    this->_assistedExcite    = assistedExcitation;
    this->_shareLayer        = shareLayer;
    this->_groupIndex        = groupIndex;
    this->_height            = height;
    this->_width             = width;
    this->_channel           = channel;
    this->_num               = num;
    this->_binary            = binary;
    this->_xnor              = xnor;
    this->_useBinOutput      = useBinOutput;
    this->_batch             = batch;
    this->_steps             = steps;
    this->_dilationX         = dilationX;
    this->_dilationY         = dilationY;
    this->_kSizeX            = kSizeX;
    this->_kSizeY            = kSizeY;
    this->_paddingX          = paddingX;
    this->_paddingY          = paddingY;
    this->_batchNorm         = batchNorm;
    this->_nWeights          = (this->_channel / groups) * num * kSizeX * kSizeY; 

    this->_useBias           = useBias;

    if(this->_useBias)
    {
        this->_nBiases       = this->_num;
    }
    else
    {
        this->_nBiases       =   0;
    }

    if(this->_shareLayer != nullptr)
    {
        if(     this->_kSizeX    != this->_shareLayer->_kSizeX ||
                this->_kSizeY    != this->_shareLayer->_kSizeY ||
                this->_nWeights != this->_shareLayer->_nWeights||
                this->_channel  != this->_shareLayer->_channel ||
                this->_num      != this->_shareLayer->_num)
        {
            throw Exception(1, "Layer size, nweights, channels or filters don't match for the share_layer", __FILE__, __LINE__);
        }

        this->_weights       = this->_shareLayer->_weights;
        this->_biases        = this->_shareLayer->_biases;
    }
    else
    {
        if(!BaseLayer::isPreviewMode)
        {
            this->_weights       = new float[static_cast<size_t>(this->_nWeights)]();
            this->_biases        = new float[static_cast<size_t>(this->_num)]();

        }
    }

    this->_outHeight         = convOutHeight();
    this->_outWidth          = convOutWidth();
    this->_outChannel        = num;      

    this->_outputNum         = this->_outHeight * this->_outWidth * this->_outChannel;
    this->_inputNum          = height * width * channel;

    this->_activation        = activation;
    this->_actParams         = actParams;

    this->_output            = new float[static_cast<size_t>(_outputNum * this->_batch)]();

    if(binary)
    {
        if(!BaseLayer::isPreviewMode)
        {
            this->_binaryWeights = new float[static_cast<size_t>(this->_nWeights)]();
            this->_cWeights      = new char[static_cast<size_t>(this->_nWeights)]();
            this->_scales        = new float[static_cast<size_t>(this->_num)]();
        }
    }

    if(xnor)
    {
        int align           = 32; 

        int srcAlign        = this->_outHeight * this->_outWidth;
        this->_bitAlign      = srcAlign + (align - srcAlign % align);
        this->_ldaAlign      = 256;

        if(!BaseLayer::isPreviewMode)
        {
            this->_binaryWeights = new float[static_cast<size_t>(this->_nWeights)]();
            this->_binaryInputs  = new float[static_cast<size_t>(this->_inputNum * this->_batch)]();
            this->_meanArr       = new float[static_cast<size_t>(this->_num)]();

            const int newCh     = this->_channel / 32;
            int rePackedISize   = newCh * this->_width * this->_height + 1;
            this->_binRePackedIn = new uint32_t[static_cast<size_t>(rePackedISize)]();

            int k               = this->_kSizeX * this->_kSizeY * this->_channel;
            int kAligned        = k + (this->_ldaAlign - k%this->_ldaAlign);
            int tBitInSize      = kAligned * this->_bitAlign / 8;
            this->_tBitInput     = new char[static_cast<size_t>(tBitInSize)]();
        }
    }

    if(batchNorm)
    {
        if(this->_shareLayer!=nullptr)
        {
            this->_scales            = this->_shareLayer->_scales;
            this->_rollMean          = this->_shareLayer->_rollMean;
            this->_rollVariance      = this->_shareLayer->_rollVariance;
        }
        else
        {
            if(!BaseLayer::isPreviewMode)
            {
                this->_scales         = new float[static_cast<size_t>(this->_num)]();
                this->_rollMean       = new float[static_cast<size_t>(this->_num)]();
                this->_rollVariance   = new float[static_cast<size_t>(this->_num)]();
            }
        }

        this->_nScales           =   num;
        this->_nRollMean         =   num;
        this->_nRollVariance     =   num;
    }

    this->_numWeights            =   static_cast<size_t>(this->_nWeights + this->_nScales + this->_nRollMean + this->_nRollVariance + this->_nBiases);

#ifndef GPU
    if(this->_activation == ActivationType::SWISH || this->_activation == ActivationType::MISH)
    {
        this->_activationInput = new float[static_cast<size_t>(this->_outputNum * totalBatch)]();
    }
#endif

    this->_workSpaceSize = getConvWorkSpaceSize();

    this->_bFlops        = (2.0f * this->_nWeights * this->_outHeight * this->_outWidth) / 1000000000.f;

    if(this->_xnor)
    {
        this->_bFlops    = this->_bFlops / 32;
    }

    if(this->_xnor && this->_useBinOutput)
    {
        this->_layerName = "ConvXB          ";
        this->_layerDetail.append("convXB");
    }
    else if(this->_xnor)
    {
        this->_layerName = "ConvX           ";
        this->_layerDetail.append("convX ");
    }
    else if(this->_shareLayer != nullptr)
    {
        this->_layerName = "ConvS           ";
        this->_layerDetail.append("convS ");
    }
    else if(this->_assistedExcite)
    {
        this->_layerName = "ConvAE          ";
        this->_layerDetail.append("convAE");
    }
    else if(this->_batchNorm)
    {
        this->_layerName = "ConvBN          ";
        this->_layerDetail.append("convBN");
    }
    else if(this->_groups == this->_num)
    {
        this->_layerName = "ConvDW          ";
        this->_layerDetail.append("convDW");
    }
    else
    {
        this->_layerName = "Conv            ";
        this->_layerDetail.append("conv  ");
    }

    char str[100];
    if(this->_groups > 1)
    {
#ifdef WIN32
        sprintf_s(str,"%5d/%4d ", this->_num, this->_groups);
#else
        sprintf(str,"%5d/%4d ", this->_num, this->_groups);
#endif
    }
    else
    {
#ifdef WIN32
        sprintf_s(str,"%5d      ", this->_num);
#else
        sprintf(str,"%5d      ", this->_num);
#endif
    }

    this->_layerDetail.append(std::string(str));

    if(this->_strideX != this->_strideY)
    {
#ifdef WIN32
        sprintf_s(str,"%2dx%2d/%2dx%2d ", this->_kSizeX, this->_kSizeY, this->_strideX, this->_strideY);
#else
        sprintf(str,"%2dx%2d/%2dx%2d ", this->_kSizeX, this->_kSizeY, this->_strideX, this->_strideY);
#endif
    }
    else
    {
        if(this->_dilationX > 1)
        {
#ifdef WIN32
            sprintf_s(str,"%2d x%2d/%2d(%1d)", this->_kSizeX, this->_kSizeY, this->_strideX, this->_dilationX);
#else
            sprintf(str,"%2d x%2d/%2d(%1d)", this->_kSizeX, this->_kSizeY, this->_strideX, this->_dilationX);
#endif
        }
        else
        {
#ifdef WIN32
            sprintf_s(str, "%2d x%2d/%2d   ", this->_kSizeX, this->_kSizeY, this->_strideX);
#else
            sprintf(str, "%2d x%2d/%2d   ", this->_kSizeX, this->_kSizeY, this->_strideX);
#endif
        }
    }

    this->_layerDetail.append(std::string(str));

#ifdef WIN32
    sprintf_s(str, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
              this->_outWidth, this->_outHeight, this->_outChannel, static_cast<double>(this->_bFlops));
#else
    sprintf(str, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
            this->_outWidth, this->_outHeight, this->_outChannel, static_cast<double>(this->_bFlops));
#endif

    this->_layerDetail.append(std::string(str));

}

ConvolutionalLayer::~ConvolutionalLayer()
{
    releaseArr(_weights);
    releaseArr(_biases);
    releaseArr(_scales);
    releaseArr(_rollMean);
    releaseArr(_rollVariance);
    releaseArr(_cWeights);
    releaseArr(_binaryInputs);
    releaseArr(_binaryWeights);
    releaseArr(_activationInput);
    releaseArr(_meanArr);
    releaseArr(_binRePackedIn);
    releaseArr(_tBitInput);
    releaseArr(_alignBitWeights);
}

int ConvolutionalLayer::convOutHeight()
{

    return (this->_height + 2 * this->_paddingY - (this->_dilationY * (this->_kSizeY - 1) + 1)) / this->_strideY + 1;
}

int ConvolutionalLayer::convOutWidth()
{

    return (this->_width + 2 * this->_paddingX - (this->_dilationX * (this->_kSizeX - 1) + 1)) / this->_strideX + 1;
}

int ConvolutionalLayer::getWorkSpaceSize32()
{
    if(this->_xnor)
    {
        int  rePackInSize   = this->_channel * this->_width * this->_height * static_cast<int>(sizeof(float));
        int  workSpaceSize  = this->_bitAlign * this->_kSizeX * this->_kSizeY * this->_channel * static_cast<int>(sizeof(float));
        if(workSpaceSize < rePackInSize)
        {
            workSpaceSize = rePackInSize;
            return workSpaceSize;
        }
    }

    return this->_outHeight * this->_outWidth * this->_kSizeX * this->_kSizeY * (this->_channel / this->_groups)*static_cast<int>(sizeof(float));
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
    float *swapV         = this->_weights;
    this->_weights       = this->_binaryWeights;
    this->_binaryWeights = swapV;
}

void ConvolutionalLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    int mOutHeight      = convOutHeight();
    int mOutWidth       = convOutWidth();
    int m       =  this->_num / this->_groups; 

    int k       =  this->_kSizeX * this->_kSizeY *this->_channel / this->_groups; 

    int n       =  mOutHeight * mOutWidth; 

    Blas::cpuFill(this->_outputNum * this->_batch, 0, this->_output, 1);

#ifdef USE_NNPACK
    struct nnp_size     nnInSize    = {static_cast<size_t>(this->_width),static_cast<size_t>(this->_height)};
    struct nnp_padding  nnInPadding = {static_cast<size_t>(this->_paddingX),static_cast<size_t>(this->_paddingX),
                                      static_cast<size_t>(this->_paddingY),static_cast<size_t>(this->_paddingY)
                                      };
    struct nnp_size     nnKSize     = {static_cast<size_t>(this->_kSizeX),static_cast<size_t>(this->_kSizeY)};
    struct nnp_size     nnStride    = {static_cast<size_t>(this->_strideX),static_cast<size_t>(this->_strideY)};
#endif

    if(this->_xnor && (!this->_alignBitWeights))
    {
        if(!this->_alignBitWeights)
        {
            binarizeWeights(this->_weights,this->_num, this->_nWeights,this->_binaryWeights);
        }
        swapBinary(); 

        cpuBinarize(netState.input, this->_channel * this->_height * this->_width * this->_batch, this->_binaryInputs);
        netState.input = this->_binaryInputs;
    }

    for (int i = 0; i < this->_batch; ++i)
    {

        for (int j = 0; j < this->_groups; ++j)
        {

            float *a    =  this->_weights + j*this->_nWeights /this->_groups;

            float *b    =  netState.workspace;

            float *c    =  this->_output + (i*this->_groups +j)*n*m;

            if(this->_xnor && this->_alignBitWeights && this->_strideX == this->_strideY)
            {
                /* TODO: */

            }
            else
            {

                float *im = netState.input + (i*this->_groups + j)*(this->_channel / this->_groups)*this->_height*this->_width;

#ifdef USE_NNPACK
                nnp_status status;
                status = nnp_convolution_inference(nnp_convolution_algorithm_implicit_gemm,
                                          nnp_convolution_transform_strategy_tuple_based,
                                          static_cast<size_t>(this->_channel/this->_groups),
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

                if(this->_kSizeX == 1 && this->_kSizeY == 1 &&  this->_strideX == 1  &&  this->_strideY == 1&& this->_paddingX == 0 && this->_paddingY == 0)
                {
                    b = im;

                }
                else
                {

                    Gemm::cpuIm2colEx(im, this->_channel/this->_groups, this->_height, this->_width, this->_kSizeX, this->_kSizeY,
                                      this->_paddingX, this->_paddingY, this->_strideX, this->_strideY, this->_dilationX, this->_dilationY,
                                      b);

                }

                Gemm::cpuGemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n, this->supportAvx&&this->supportFma);
#endif

            }

        }

    }

    if(this->_batchNorm==1)
    {

        for (int b = 0; b < this->_batch; ++b)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int c = 0; c < this->_outChannel; ++c)
            {
#ifdef USE_ARM
                for (int i = 0; i < this->_outHeight*this->_outWidth; ++i)
                {
                    int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i;

                    this->_output[index]  = this->_scales[c]*(this->_output[index] - this->_rollMean[c])/sqrt(this->_rollVariance[c] + 0.00001f) + this->_biases[c];
                }
#endif

#ifdef USE_X86
                if(this->supportAvx)
                {
                    int i = 0;
                    for (; i < (this->_outHeight*this->_outWidth)/8; ++i)
                    {

                        int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i*8;

                        __m256 mScale;
                        __m256 mInput;
                        __m256 mMean;
                        __m256 mVariance;
                        __m256 mEsp;
                        __m256 mBias;
                        __m256 mResult1;
                        __m256 mResult2;

                        mScale      =   _mm256_set1_ps(this->_scales[c]);
                        mInput      =   _mm256_loadu_ps(this->_output+index);
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
                        int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + j;
                        this->_output[index]  = this->_scales[c]*(this->_output[index] - this->_rollMean[c])/sqrt(this->_rollVariance[c] + 0.00001f) + this->_biases[c];
                    }
                }
                else
                {
                    for (int i = 0; i < this->_outHeight*this->_outWidth; ++i)
                    {
                        int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i;

                        this->_output[index]  = this->_scales[c]*(this->_output[index] - this->_rollMean[c])/sqrt(this->_rollVariance[c] + 0.00001f) + this->_biases[c];
                    }
                }
#endif
            }
        }

    }
    else
    {
        if(_useBias == 1)
            addBias(this->_output, this->_biases, this->_batch, this->_num, mOutHeight*mOutWidth);
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

    if(this->_binary || this->_xnor)
    {
        swapBinary();
    }

    auto so = std::chrono::system_clock::now();

    this->_forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

void ConvolutionalLayer::loadAllWeigths(std::vector<float> &weights)
{
    if(weights.size() != this->_numWeights)
    {
        throw Exception(1,"Conv weights load err. needed : " + std::to_string(this->_numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__);
    }

    loadWeights(weights.data(), _nWeights);

    if(this->_batchNorm)
    {
        loadScales(weights.data() + _nWeights, _nScales);
        loadBias(weights.data() + _nWeights + _nScales, _nBiases);
        loadRollMean(weights.data() + _nWeights + _nScales + _nBiases, _nRollMean);
        loadRollVariance(weights.data() + _nWeights + _nScales + _nBiases + _nRollMean, _nRollVariance);
    }
    else
    {
        if(_useBias==1)
        {
            loadBias(weights.data() + _nWeights, _nBiases);
        }
    }
}

void ConvolutionalLayer::loadScales(float * const &weights, const int &len)
{
    if(len != this->_nScales)
    {
        throw Exception(1, "load scales data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, weights, 1, this->_scales,1);
}

void ConvolutionalLayer::loadBias(float * const &bias, const int &len)
{
    if(len != this->_nBiases)
    {
        throw Exception(1, "load bias data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, bias, 1, this->_biases,1);
}

void ConvolutionalLayer::loadWeights(float * const &weights, const int &len)
{
    if(len != this->_nWeights)
    {
        throw Exception(1, "load weights data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, weights, 1, this->_weights,1);
}

void ConvolutionalLayer::loadRollMean(float * const &rollMean, const int &len)
{
    if(len != this->_nRollMean)
    {
        throw Exception(1, "load roll mean data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, rollMean, 1, this->_rollMean,1);
}

void ConvolutionalLayer::loadRollVariance(float * const &rollVariance, const int &len)
{
    if(len != this->_nRollVariance)
    {
        throw Exception(1, "load roll variance data len error ",__FILE__,__LINE__);
    }
    Blas::cpuCopy(len, rollVariance, 1, this->_rollVariance,1);
}

float *ConvolutionalLayer::getWeights() const
{
    return _weights;
}

float *ConvolutionalLayer::getBiases() const
{
    return _biases;
}

float *ConvolutionalLayer::getScales() const
{
    return _scales;
}

float *ConvolutionalLayer::getRollMean() const
{
    return _rollMean;
}

float *ConvolutionalLayer::getRollVariance() const
{
    return _rollVariance;
}

char *ConvolutionalLayer::getCWeights() const
{
    return _cWeights;
}

float *ConvolutionalLayer::getBinaryInputs() const
{
    return _binaryInputs;
}

float *ConvolutionalLayer::getBinaryWeights() const
{
    return _binaryWeights;
}

float *ConvolutionalLayer::getActivationInput() const
{
    return _activationInput;
}

float *ConvolutionalLayer::getMeanArr() const
{
    return _meanArr;
}

uint32_t *ConvolutionalLayer::getBinRePackedIn() const
{
    return _binRePackedIn;
}

char *ConvolutionalLayer::getTBitInput() const
{
    return _tBitInput;
}

char *ConvolutionalLayer::getAlignBitWeights() const
{
    return _alignBitWeights;
}

int ConvolutionalLayer::getBitAlign() const
{
    return _bitAlign;
}

int ConvolutionalLayer::getLdaAlign() const
{
    return _ldaAlign;
}

int ConvolutionalLayer::getUseBias() const
{
    return _useBias;
}

int ConvolutionalLayer::getNScales() const
{
    return _nScales;
}

int ConvolutionalLayer::getNRollMean() const
{
    return _nRollMean;
}

int ConvolutionalLayer::getNRollVariance() const
{
    return _nRollVariance;
}

int ConvolutionalLayer::getNBiases() const
{
    return _nBiases;
}

int ConvolutionalLayer::getNWeights() const
{
    return _nWeights;
}

int ConvolutionalLayer::getGroups() const
{
    return _groups;
}

int ConvolutionalLayer::getGroupIndex() const
{
    return _groupIndex;
}

int ConvolutionalLayer::getXnor() const
{
    return _xnor;
}

int ConvolutionalLayer::getBinary() const
{
    return _binary;
}

int ConvolutionalLayer::getUseBinOutput() const
{
    return _useBinOutput;
}

int ConvolutionalLayer::getSteps() const
{
    return _steps;
}

int ConvolutionalLayer::getAntialiasing() const
{
    return _antialiasing;
}

int ConvolutionalLayer::getAssistedExcite() const
{
    return _assistedExcite;
}

int ConvolutionalLayer::getKSizeX() const
{
    return _kSizeX;
}

int ConvolutionalLayer::getKSizeY() const
{
    return _kSizeY;
}

int ConvolutionalLayer::getStride() const
{
    return _stride;
}

int ConvolutionalLayer::getStrideX() const
{
    return _strideX;
}

int ConvolutionalLayer::getStrideY() const
{
    return _strideY;
}

int ConvolutionalLayer::getPaddingX() const
{
    return _paddingX;
}

int ConvolutionalLayer::getPaddingY() const
{
    return _paddingY;
}

int ConvolutionalLayer::getDilationX() const
{
    return _dilationX;
}

int ConvolutionalLayer::getDilationY() const
{
    return _dilationY;
}

int ConvolutionalLayer::getBatchNorm() const
{
    return _batchNorm;
}

}
