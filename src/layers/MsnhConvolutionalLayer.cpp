#include "Msnhnet/layers/MsnhConvolutionalLayer.h"

namespace Msnhnet
{
ConvolutionalLayer::ConvolutionalLayer(const int &batch, const int &steps, const int &height, const int &width, const int &channel, const int &num,
                                       const int &groups, const int &kSizeX, const int &kSizeY, const int &strideX, const int &strideY, const int &dilationX, const int &dilationY,
                                       const int &paddingX, const int &paddingY, ActivationType activation, const std::vector<float> &actParams, const int &batchNorm, const int &useBias, const int &binary, const int &xnor, const int &useBinOutput, const int &groupIndex, const int &antialiasing,
                                       ConvolutionalLayer * const &shareLayer, const int &assistedExcitation, const int &deform)
{
    (void) deform;
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
            throw Exception(1, "Layer size, nweights, channels or filters don't match for the share_layer", __FILE__, __LINE__, __FUNCTION__);
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
#ifdef USE_GPU
            this->_gpuWeights    = Cuda::makeCudaArray(this->_weights, this->_nWeights);
            this->_gpuBiases     = Cuda::makeCudaArray(this->_biases , this->_num);
#ifdef USE_CUDNN
            if (useFp16)
            {
                this->_gpuWeightsFp16 = Cuda::makeCudaArray(this->_weights, this->_nWeights);
            }
#endif 
#endif
        }
    }

    this->_outHeight         = convOutHeight();
    this->_outWidth          = convOutWidth();
    this->_outChannel        = num;      

    this->_outputNum         = this->_outHeight * this->_outWidth * this->_outChannel;
    this->_inputNum          = height * width * channel;

    this->_activation        = activation;
    this->_actParams         = actParams;

    if(!BaseLayer::isPreviewMode)
    {
        this->_output            = new float[static_cast<size_t>(_outputNum * this->_batch)]();
#ifdef USE_GPU
        this->_gpuOutput         = Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
#ifdef USE_CUDNN
        if (useFp16)
        {
            this->_gpuOutputFp16 = Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
        }
#endif 

#endif
    }

    if(binary)
    {
        if(!BaseLayer::isPreviewMode)
        {
            this->_binaryWeights = new float[static_cast<size_t>(this->_nWeights)]();
            this->_cWeights      = new char[static_cast<size_t>(this->_nWeights)]();
            this->_scales        = new float[static_cast<size_t>(this->_num)]();

#ifdef USE_GPU

#endif
        }
    }

    if(xnor)
    {
        int align            = 32; 

        int srcAlign         = this->_outHeight * this->_outWidth;
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
#ifdef USE_GPU

#endif
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
#ifdef USE_GPU
                this->_gpuScales      = Cuda::makeCudaArray(this->_scales, this->_num);
                this->_gpuRollMean    = Cuda::makeCudaArray(this->_rollMean, this->_num);
                this->_gpuRollVariance= Cuda::makeCudaArray(this->_rollVariance, this->_num);
#endif

            }
        }

        this->_nScales           =   num;
        this->_nRollMean         =   num;
        this->_nRollVariance     =   num;
    }

    this->_numWeights            =   static_cast<size_t>(this->_nWeights + this->_nScales + this->_nRollMean + this->_nRollVariance + this->_nBiases);

#ifdef USE_GPU
#ifdef USE_CUDNN

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->_inputDesc16));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->_inputDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, this->_batch, this->_channel, this->_height, this->_width));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->_outputDesc16));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->_outputDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, this->_batch, this->_outChannel, this->_outHeight, this->_outWidth));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&this->_weightDesc16));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(this->_weightDesc16, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, this->_num, this->_channel/this->_groups, this->_kSizeY, this->_kSizeX));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->_inputDesc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->_inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->_batch, this->_channel, this->_height, this->_width));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->_outputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->_outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->_batch, this->_outChannel, this->_outHeight, this->_outWidth));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&this->_weightDesc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(this->_weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, this->_num, this->_channel/this->_groups, this->_kSizeY, this->_kSizeX));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&this->_convDesc));

    CUDNN_CHECK(cudnnSetConvolutionGroupCount(this->_convDesc,this->_groups));
    CUDNN_CHECK(cudnnSetConvolutionMathType(this->_convDesc, CUDNN_TENSOR_OP_MATH));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(this->_convDesc, this->_paddingY, this->_paddingX, this->_strideY, this->_strideX,
                                                this->_dilationY, this->_dilationX, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    this->_fwAlgo16 = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(Cuda::getCudnnHandle(), this->_inputDesc, this->_weightDesc, this->_convDesc, this->_outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0,

                                                    &this->_fwAlgo));

#endif
#endif

    this->_workSpaceSize = getConvWorkSpaceSize();

    this->_inputSpaceSize = _inputNum;

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
    releaseArr(_meanArr);
    releaseArr(_binRePackedIn);
    releaseArr(_tBitInput);
    releaseArr(_alignBitWeights);

#ifdef USE_GPU
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_outputDesc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(_weightDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_inputDesc16));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_outputDesc16));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(_weightDesc16));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(_convDesc));
    if (useFp16)
    {
        Cuda::freeCuda(_gpuWeightsFp16);
        Cuda::freeCuda(_gpuOutputFp16);
    }
#endif
    Cuda::freeCuda(_gpuWeights);
    Cuda::freeCuda(_gpuBiases);
    Cuda::freeCuda(_gpuScales);
    Cuda::freeCuda(_gpuRollMean);
    Cuda::freeCuda(_gpuRollVariance);

#endif
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
    size_t space = 0;
#ifdef USE_GPU
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(Cuda::getCudnnHandle(), this->_inputDesc, this->_weightDesc, this->_convDesc,
                                                        this->_outputDesc, this->_fwAlgo, &space));
#endif
#endif

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

    size_t space1 = this->_outHeight * this->_outWidth * this->_kSizeX * this->_kSizeY * (this->_channel / this->_groups)*static_cast<int>(sizeof(float));

    return (space1>space)?space1:space;
}

int ConvolutionalLayer::getWorkSpaceSize16()
{
#ifdef USE_GPU
#ifdef USE_CUDNN
    if(useFp16)
    {
        size_t space = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(Cuda::getCudnnHandle(), this->_inputDesc16, this->_weightDesc16, this->_convDesc,
                                                            this->_outputDesc16, this->_fwAlgo16, &space));
        return space;
    }
#endif
#endif
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
    /* TODO */
    float *swapV            = this->_weights;
    this->_weights          = this->_binaryWeights;
    this->_binaryWeights    = swapV;
#ifdef USE_GPU

#endif
}

void ConvolutionalLayer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();

    int m       =  this->_num / this->_groups; 

    int k       =  this->_kSizeX * this->_kSizeY *this->_channel / this->_groups; 

    int n       =  this->_outHeight * this->_outWidth; 

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
                    throw Exception(1,"NNPack error, code : "+std::to_string(status),__FILE__,__LINE__, __FUNCTION__);
                }
#else

#ifdef USE_ARM
                if(this->_kSizeX == 3 && this->_kSizeY == 3 && this->_strideX == 1 && this->_strideX == 1&& this->_paddingX == 0 && this->_paddingY == 0)
                {

#ifdef __arrch64__
                    goto TempARRCH64;
#endif
                    ConvolutionalLayerArm3x3s1::conv3x3s1Neon(im, this->_width, this->_height, this->_channel, a, c, this->_outWidth, this->_outHeight, this->_outChannel);
                }
                else if(this->_kSizeX == 3 && this->_kSizeY == 3 && this->_strideX == 2 && this->_strideX == 2&& this->_paddingX == 0 && this->_paddingY == 0)
                {
#ifdef __arrch64__
                    goto TempARRCH64;
#endif
                    ConvolutionalLayerArm3x3s2::conv3x3s2Neon(im, this->_width, this->_height, this->_channel, a, c, this->_outWidth, this->_outHeight, this->_outChannel);
                }
                else
                {
#endif
TempARRCH64:

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
#ifdef USE_ARM
                }
#endif

#endif
            }

        }

    }

    if(this->_batchNorm==1)
    {
        for (int b = 0; b < this->_batch; ++b)
        {
#ifdef USE_ARM
#ifdef USE_NEON
        int step = b*this->_outChannel*this->_outHeight*this->_outWidth;
        BatchNormLayerArm::BatchNorm(this->_output + step,
                                     this->_width,
                                     this->_height,
                                     this->_channel,
                                     this->_output + step,
                                     this->_scales,
                                     this->_rollMean,
                                     this->_rollVariance,
                                     this->_biases
                                     );
#else
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int c = 0; c < this->_outChannel; ++c)
            {
                float sqrtVal   = sqrt(this->_rollVariance[c] + 0.00001f);
                float scaleSqrt = this->_scales[c]/sqrtVal;
                float meanSqrt  = -this->_scales[c]*this->_rollMean[c]/sqrtVal;
                for (int i = 0; i < this->_outHeight*this->_outWidth; ++i)
                {
                    int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i;

                    this->_output[index]  = scaleSqrt*this->_output[index] + meanSqrt + this->_biases[c];
                }
            }
#endif
#endif

#ifdef USE_X86
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int c = 0; c < this->_outChannel; ++c)
            {
                float sqrtVal   = sqrt(this->_rollVariance[c] + 0.00001f);
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
                        mInput      =   _mm256_loadu_ps(this->_output+index);
                        mMeanSqrt   =   _mm256_set1_ps(meanSqrt);
                        mBias       =   _mm256_set1_ps(this->_biases[c]);
                        mResult     =   _mm256_mul_ps(mScaleSqrt, mInput);
                        mResult     =   _mm256_add_ps(mResult, mMeanSqrt);
                        mResult     =   _mm256_add_ps(mResult, mBias);

                        _mm256_storeu_ps(this->_output+index, mResult);

                    }

                    for (int j = (this->_outHeight*this->_outWidth)/8*8; j < this->_outHeight*this->_outWidth; ++j)
                    {
                        int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + j;
                        this->_output[index]  = scaleSqrt*this->_output[index] + meanSqrt + this->_biases[c];
                    }
                }
                else
                {

                    for (int i = 0; i < this->_outHeight*this->_outWidth; ++i)
                    {
                        int index = b*this->_outChannel*this->_outHeight*this->_outWidth + c*this->_outHeight*this->_outWidth + i;
                        this->_output[index]  = scaleSqrt*this->_output[index] + meanSqrt + this->_biases[c];
                    }
                }
            }
#endif
        }

    }
    else
    {
        if(_useBias == 1)
            addBias(this->_output, this->_biases, this->_batch, this->_num, this->_outHeight * this->_outWidth);
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

    /* xnor and binary TODO: */
    if(this->_binary || this->_xnor)
    {
        swapBinary();
    }

    this->_forwardTime =   TimeUtil::getElapsedTime(st);

}

#ifdef USE_GPU
void ConvolutionalLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

#ifdef USE_CUDNN
    if(!onlyUseCuda)
    {
        float a = 1.f;
        float b = 0;

        if(useFp16 && (this->_num%8==0) && (this->_channel%8==0) && this->_groups == 1)
        {
            Cuda::fp32ToFp16(netState.input, netState.inputNum, netState.gpuInputFp16);
            Cuda::fp32ToFp16(this->_gpuWeights, this->_nWeights, this->_gpuWeightsFp16);

            CUDNN_CHECK(cudnnConvolutionForward(Cuda::getCudnnHandle(),
                                                &a,
                                                this->_inputDesc16, netState.gpuInputFp16,
                                                this->_weightDesc16, this->_gpuWeightsFp16,
                                                this->_convDesc, this->_fwAlgo16,
                                                netState.gpuWorkspace, this->_workSpaceSize,
                                                &b,
                                                this->_outputDesc16, this->_gpuOutputFp16));

            Cuda::fp16ToFp32(this->_gpuOutputFp16, this->_outputNum, this->_gpuOutput);
        }
        else
        {
            CUDNN_CHECK(cudnnConvolutionForward(Cuda::getCudnnHandle(),
                                                &a,
                                                this->_inputDesc, netState.input,
                                                this->_weightDesc, this->_gpuWeights,
                                                this->_convDesc, this->_fwAlgo,
                                                netState.gpuWorkspace, this->_workSpaceSize,
                                                &b,
                                                this->_outputDesc, this->_gpuOutput));
        }
    }
    else
    {
        int m       =  this->_num / this->_groups; 

        int k       =  this->_kSizeX * this->_kSizeY *this->_channel / this->_groups; 

        int n       =  this->_outHeight * this->_outWidth; 

        BlasGPU::gpuFill(this->_outputNum * this->_batch, 0, this->_gpuOutput, 1);

        for (int i = 0; i < this->_batch; ++i)
        {

            for (int j = 0; j < this->_groups; ++j)
            {

                float *a    =  this->_gpuWeights + j*this->_nWeights /this->_groups;

                float *b    =  netState.gpuWorkspace;

                float *c    =  this->_gpuOutput + (i*this->_groups +j)*n*m;

                if(this->_xnor && this->_alignBitWeights && this->_strideX == this->_strideY)
                {
                    /* TODO */
                }
                else
                {

                    float *im = netState.input + (i*this->_groups + j)*(this->_channel / this->_groups)*this->_height*this->_width;

                    if(this->_kSizeX == 1 && this->_kSizeY == 1 &&  this->_strideX == 1  &&  this->_strideY == 1&& this->_paddingX == 0 && this->_paddingY == 0)
                    {
                        b = im;
                    }
                    else
                    {

                        GemmGPU::gpuIm2ColEx(im, this->_channel/this->_groups, this->_height, this->_width, this->_kSizeX, this->_kSizeY,
                                             this->_paddingX, this->_paddingY, this->_strideX, this->_strideY, this->_dilationX, this->_dilationY,
                                             b);
                    }

                    GemmGPU::gpuGemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
                }

            }

        }
    }
#else
    int m       =  this->_num / this->_groups; 

    int k       =  this->_kSizeX * this->_kSizeY *this->_channel / this->_groups; 

    int n       =  this->_outHeight * this->_outWidth; 

    BlasGPU::gpuFill(this->_outputNum * this->_batch, 0, this->_gpuOutput, 1);

    for (int i = 0; i < this->_batch; ++i)
    {

        for (int j = 0; j < this->_groups; ++j)
        {

            float *a    =  this->_gpuWeights + j*this->_nWeights /this->_groups;

            float *b    =  netState.gpuWorkspace;

            float *c    =  this->_gpuOutput + (i*this->_groups +j)*n*m;

            if(this->_xnor && this->_alignBitWeights && this->_strideX == this->_strideY)
            {
                /* TODO */
            }
            else
            {

                float *im = netState.input + (i*this->_groups + j)*(this->_channel / this->_groups)*this->_height*this->_width;

                if(this->_kSizeX == 1 && this->_kSizeY == 1 &&  this->_strideX == 1  &&  this->_strideY == 1&& this->_paddingX == 0 && this->_paddingY == 0)
                {
                    b = im;
                }
                else
                {

                    GemmGPU::gpuIm2ColEx(im, this->_channel/this->_groups, this->_height, this->_width, this->_kSizeX, this->_kSizeY,
                                         this->_paddingX, this->_paddingY, this->_strideX, this->_strideY, this->_dilationX, this->_dilationY,
                                         b);
                }

                GemmGPU::gpuGemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
            }

        }

    }
#endif

    if(this->_batchNorm == 1)
    {

        ConvolutionalLayerGPU::convBn(this->_batch, this->_outChannel, this->_outHeight, this->_outWidth, this->_gpuScales,
                                      this->_gpuRollMean, this->_gpuRollVariance, this->_gpuBiases, this->_gpuOutput
                                      );
    }
    else
    {
        if(this->_useBias)
        {
            BlasGPU::gpuAddBias(this->_gpuOutput, this->_gpuBiases, this->_batch, this->_outChannel, this->_outHeight*this->_outWidth);
        }
    }

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

    /* xnor and binary TODO: */
    if(this->_binary || this->_xnor)
    {
        swapBinary();
    }

    if(netState.fixNan==1)
    {
        BlasGPU::gpuFixNanAndInf(this->_gpuOutput, this->_outputNum*this->_batch);
    }

    this->recordCudaStop();
}
#endif

void ConvolutionalLayer::loadAllWeigths(std::vector<float> &weights)
{
    if(weights.size() != this->_numWeights)
    {
        throw Exception(1,"Conv weights load err. needed : " + std::to_string(this->_numWeights) + " given : " +  std::to_string(weights.size()), __FILE__, __LINE__, __FUNCTION__);
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
        if(this->_useBias==1)
        {
            loadBias(weights.data() + _nWeights, _nBiases);
        }
    }

#ifdef USE_GPU
    Cuda::pushCudaArray(this->_gpuWeights, this->_weights, this->_nWeights);
    if(this->_batchNorm)
    {
        Cuda::pushCudaArray(this->_gpuScales       , this->_scales       , this->_nScales      );
        Cuda::pushCudaArray(this->_gpuBiases       , this->_biases       , this->_nBiases      );
        Cuda::pushCudaArray(this->_gpuRollMean     , this->_rollMean     , this->_nRollMean    );
        Cuda::pushCudaArray(this->_gpuRollVariance , this->_rollVariance, this->_nRollVariance);
    }
    else
    {
        if(this->_useBias == 1)
        {
            Cuda::pushCudaArray(this->_gpuBiases, this->_biases, this->_nBiases);
        }
    }
#endif
}

void ConvolutionalLayer::loadScales(float * const &weights, const int &len)
{
    if(len != this->_nScales)
    {
        throw Exception(1, "load scales data len error ",__FILE__,__LINE__, __FUNCTION__);
    }
    Blas::cpuCopy(len, weights, 1, this->_scales,1);
}

void ConvolutionalLayer::loadBias(float * const &bias, const int &len)
{
    if(len != this->_nBiases)
    {
        throw Exception(1, "load bias data len error ",__FILE__,__LINE__, __FUNCTION__);
    }
    Blas::cpuCopy(len, bias, 1, this->_biases,1);
}

void ConvolutionalLayer::loadWeights(float * const &weights, const int &len)
{
    if(len != this->_nWeights)
    {
        throw Exception(1, "load weights data len error ",__FILE__,__LINE__, __FUNCTION__);
    }
    Blas::cpuCopy(len, weights, 1, this->_weights,1);
}

void ConvolutionalLayer::loadRollMean(float * const &rollMean, const int &len)
{
    if(len != this->_nRollMean)
    {
        throw Exception(1, "load roll mean data len error ",__FILE__,__LINE__, __FUNCTION__);
    }
    Blas::cpuCopy(len, rollMean, 1, this->_rollMean,1);
}

void ConvolutionalLayer::loadRollVariance(float * const &rollVariance, const int &len)
{
    if(len != this->_nRollVariance)
    {
        throw Exception(1, "load roll variance data len error ",__FILE__,__LINE__, __FUNCTION__);
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
