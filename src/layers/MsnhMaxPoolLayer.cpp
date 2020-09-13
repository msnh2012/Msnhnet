#include "Msnhnet/layers/MsnhMaxPoolLayer.h"

namespace Msnhnet
{
MaxPoolLayer::MaxPoolLayer(const int &batch,   const int &height, const int &width, const int &channel, const int &kSizeX,  const int &kSizeY,
                           const int &strideX, const int &strideY, const int &paddingX, const int &paddingY, const int &maxPoolDepth,
                           const int &outChannelsMp, const int& ceilMode,  const int &antialiasing)
{
    this->_type           = LayerType::MAXPOOL;
    this->_layerName      = "MaxPool         ";

    this->_batch          = batch;
    this->_height         = height;
    this->_width          = width;
    this->_channel        = channel;

    this->_paddingX       = paddingX;
    this->_paddingY       = paddingY;

    this->_ceilMode       = ceilMode;

    this->_maxPoolDepth   = maxPoolDepth;
    this->_outChannelsMp  = outChannelsMp;   

    this->_antialiasing   = antialiasing;

    this->_kSizeX         = kSizeX;  

    this->_kSizeY         = kSizeY;  

    this->_stride         = strideX; 

    this->_strideX        = strideX;

    this->_strideY        = strideY;

    if(maxPoolDepth)
    {
        this->_outChannel = outChannelsMp;
        this->_outWidth   = this->_width;
        this->_outHeight  = this->_height;
    }
    else
    {

        if(this->_ceilMode == 1)
        {
            int tmpW = (width  + paddingX*2 - kSizeX) % strideX; 

            int tmpH = (height + paddingY*2 - kSizeY) % strideY; 

            if(tmpW >= kSizeX)
            {
                throw Exception(1,"maxpool padding error ", __FILE__, __LINE__, __FUNCTION__);
            }

            if(tmpH >= kSizeY)
            {
                throw Exception(1,"maxpool padding error ", __FILE__, __LINE__, __FUNCTION__);
            }

            if(tmpW <= paddingX)
            {
                this->_outWidth   = (width  + paddingX*2 - kSizeX) / strideX + 1; 

            }
            else
            {
                this->_outWidth   = (width  + paddingX*2 - kSizeX) / strideX + 2; 

            }

            if(tmpH <= paddingY)
            {
                this->_outHeight  = (height + paddingY*2 - kSizeY) / strideY + 1; 

            }
            else
            {
                this->_outHeight  = (height + paddingY*2 - kSizeY) / strideY + 2; 

            }
        }
        else if(this->_ceilMode == 0)
        {
            this->_outWidth   = (width  + paddingX*2 - kSizeX) / strideX + 1; 

            this->_outHeight  = (height + paddingY*2 - kSizeY) / strideY + 1; 

        }
        else
        {
            this->_outWidth   = (width  + paddingX - kSizeX) / strideX + 1; 

            this->_outHeight  = (height + paddingY - kSizeY) / strideY + 1; 

        }
        this->_outChannel = channel;                                 

    }

    this->_outputNum      =  this->_outHeight * this->_outWidth * this->_outChannel; 

    this->_inputNum       =  height * width * channel; 

#ifdef USE_GPU
#ifdef USE_CUDNN

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->_inputDesc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->_inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->_batch, this->_channel, this->_height, this->_width));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->_outputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->_outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->_batch, this->_outChannel, this->_outHeight, this->_outWidth));

    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&this->_maxPoolDesc));

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(this->_maxPoolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,  

                                            this->_kSizeY, this->_kSizeX, this->_paddingY, this->_paddingX, this->_strideY,this->_strideX));
#endif
#endif

    this->_bFlops            = (this->_kSizeX*this->_kSizeY* this->_channel*this->_outHeight*this->_outWidth)/ 1000000000.f;

    this->_maxOutputNum  = this->_batch*this->_outputNum;

    char msg[100];

    if(maxPoolDepth)
    {
#ifdef WIN32
        sprintf_s(msg, "MaxPooldepth      %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  this->_kSizeX, this->_kSizeY, this->_strideX, this->_width, this->_height, this->_channel,
                  this->_outWidth, this->_outHeight,this->_outChannel,this->_bFlops);
#else
        sprintf(msg, "max-depth         %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                this->_kSizeX, this->_kSizeY, this->_strideX, this->_width, this->_height, this->_channel,
                this->_outWidth, this->_outHeight,this->_outChannel,this->_bFlops);
#endif
    }

    if(strideX == strideY)
    {
#ifdef WIN32
        sprintf_s(msg, "MaxPool           %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  this->_kSizeX, this->_kSizeY, this->_strideX, this->_width, this->_height, this->_channel,
                  this->_outWidth, this->_outHeight,this->_outChannel,this->_bFlops);
#else
        sprintf(msg, "max               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                this->_kSizeX, this->_kSizeY, this->_strideX, this->_width, this->_height, this->_channel,
                this->_outWidth, this->_outHeight,this->_outChannel,this->_bFlops);
#endif
    }
    else
    {
#ifdef WIN32
        sprintf_s(msg, "MaxPool          %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  this->_kSizeX, this->_kSizeY, this->_strideX,this->_strideY, this->_width, this->_height,
                  this->_channel, this->_outWidth, this->_outHeight,this->_outChannel,this->_bFlops);
#else
        sprintf(msg, "max              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                this->_kSizeX, this->_kSizeY, this->_strideX,this->_strideY, this->_width, this->_height,
                this->_channel, this->_outWidth, this->_outHeight,this->_outChannel,this->_bFlops);
#endif
    }

    this->_layerDetail       = msg;

}

void MaxPoolLayer::forward(NetworkState &netState)
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

    if(this->_maxPoolDepth)
    {
        for(int b=0; b<this->_batch; ++b)                    

        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif                                                      

            for(int i=0; i<this->_height; i++)               

            {
                for(int j=0; j<this->_width; ++j)            

                {
                    for(int g=0; g<this->_outChannel; ++g)   

                    {
                        int outIndex = j + this->_width*(i + this->_height*(g + this->_outChannel*b));
                        float max    = -FLT_MAX;

                        for(int k=g; k<this->_channel; k+=this->_outChannel)
                        {
                            int inIndex = j + this->_width*(i + this->_height*(k + this->_channel*b));
                            float val   = layerInput[inIndex];

                            max         = (val > max)? val:max;
                        }

                        layerOutput[outIndex] = max;
                    }

                }

            }
        }
        this->_forwardTime = TimeUtil::getElapsedTime(st);
        return;
    }
#ifdef USE_X86
    if((this->_strideX == this->_strideY) && supportAvx )
    {
        forwardAvx(layerInput,layerOutput,this->_kSizeX, this->_kSizeY, this->_width,this->_height,this->_outWidth,
                   this->_outHeight,this->_channel,this->_paddingX, this->_paddingY,this->_stride,this->_batch);
    }
    else
#endif
    {

        int widthOffset  =     -this->_paddingX;
        int heightOffset =     -this->_paddingY;

        int mHeight         =   this->_outHeight;
        int mWidth          =   this->_outWidth;

        int mChannel        =   this->_channel;

        for(int b=0; b<this->_batch; ++b)                

        {
            for(int k=0; k<mChannel; ++k)               

            {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
                for(int i=0; i<mHeight; ++i)            

                {
                    for(int j=0; j<mWidth; ++j)         

                    {

                        int outIndex = j + mWidth*(i + mHeight*(k + _channel*b));
                        float max    = -FLT_MAX;

                        for(int n=0; n<this->_kSizeY; ++n)
                        {
                            for(int m=0; m<this->_kSizeX; ++m)
                            {

                                int curHeight =  heightOffset + i*this->_strideY + n;

                                int curWidth  =  widthOffset  + j*this->_strideX + m;

                                int index     =  curWidth + this->_width*(curHeight + this->_height*(k + b*this->_channel));

                                bool valid    =  (curHeight >=0 && curHeight < this->_height &&
                                                  curWidth  >=0 && curWidth  < this->_width);

                                float value   =  (valid)? layerInput[index] : -FLT_MAX;

                                max           =  (value > max) ? value : max;
                            }
                        }

                        layerOutput[outIndex] = max;
                    }
                }
            }
        }
    }

    this->_forwardTime = TimeUtil::getElapsedTime(st);

}

void MaxPoolLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {

            if(!BaseLayer::onlyUseGpu) 

            {
                this->_output         = new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
            }
#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuOutput      = Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }
    this->_memReUse         =  0;
}

#ifdef USE_GPU
void MaxPoolLayer::forwardGPU(NetworkState &netState)
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

    if(this->_maxPoolDepth)
    {
        MaxPoolLayerGPU::forwardDepthGPU(this->_width, this->_height, this->_channel, this->_outWidth, this->_outHeight, this->_outChannel, this->_batch, layerGpuInput, layerGpuOutput);
    }
    else
    {
#ifdef USE_CUDNN
        if(!onlyUseCuda)
        {
            float a = 1.f;
            float b = 0;
            CUDNN_CHECK(cudnnPoolingForward(Cuda::getCudnnHandle(), this->_maxPoolDesc, &a,
                                            this->_inputDesc, layerGpuInput,
                                            &b,
                                            this->_outputDesc, layerGpuOutput));
        }
        else
        {
            MaxPoolLayerGPU::forwardNormalGPU(this->_width,this->_height,this->_channel,
                                              this->_outWidth, this->_outHeight, this->_outChannel,
                                              this->_strideX, this->_strideY,
                                              this->_kSizeX, this->_kSizeY,
                                              this->_paddingX, this->_paddingY,
                                              this->_batch,
                                              layerGpuInput,
                                              layerGpuOutput
                                              );
        }

#else
        MaxPoolLayerGPU::forwardNormalGPU(this->_width,this->_height,this->_channel,
                                          this->_outWidth, this->_outHeight, this->_outChannel,
                                          this->_strideX, this->_strideY,
                                          this->_kSizeX, this->_kSizeY,
                                          this->_paddingX, this->_paddingY,
                                          this->_batch,
                                          layerGpuInput,
                                          layerGpuOutput
                                          );
#endif
    }

    this->recordCudaStop();
}
#endif

#ifdef USE_X86
void MaxPoolLayer::forwardAvx(float *const &src, float *const &dst, const int &kSizeX, const int &kSizeY,
                              const int &width, const int &height, const int &outWidth, const int &outHeight,
                              const int &channel,const int &paddingX,const int &paddingY,const int &stride, const int &batch)
{

    int widthOffset  =     -paddingX ;
    int heightOffset =     -paddingY ;

    for(int b=0; b<batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int k=0; k<channel; ++k)
        {
            for(int i=0; i<outHeight; ++i)
            {
                int j = 0;
                if(stride == 1)  

                {
                    for(j=0; j<outWidth - 8 -(kSizeX - 1); j+=8)  

                    {
                        int outIndex = j + outWidth*(i + outHeight*(k + channel*b));
                        __m256 max256    = _mm256_set1_ps(-FLT_MAX);
                        for(int n=0; n<kSizeY; ++n)  

                        {
                            for(int m=0; m<kSizeX; ++m) 

                            {
                                int curHeight = heightOffset + i*stride + n;
                                int curWidth  = widthOffset  + j*stride + m;
                                int index     = curWidth +width*(curHeight + height*(k + b*channel));

                                int valid     = (curHeight >=0 && curHeight < height && curWidth>=0 && curWidth<width );

                                if(!valid) continue;

                                __m256 src256 = _mm256_loadu_ps(&src[index]);

                                max256        = _mm256_max_ps(src256,max256);
                            }
                        }

                        _mm256_storeu_ps(&dst[outIndex], max256);

                    }
                }
                else if(kSizeX == 2 && kSizeX == 2 && stride == 2)  

                {
                    for(j=0; j<outWidth-4; j+=4)
                    {
                        int outIndex = j + outWidth * (i + outHeight*(k + channel*b));

                        __m128 max128 = _mm_set1_ps(-FLT_MAX);

                        for(int n=0; n<kSizeX; ++n)
                        {
                            int m = 0;
                            {
                                int curHeight  = heightOffset + i * stride + n;
                                int curWidth   = widthOffset  + j * stride + m;

                                int index      = curWidth + width*(curHeight + height*(k + b*channel));
                                int valid      = (curHeight >=0 && curHeight < height && curWidth>=0 && curWidth<width );

                                if (!valid) continue;

                                __m256 src256   = _mm256_loadu_ps(&src[index]);

                                __m256 src256_2 = _mm256_permute_ps(src256, 0b10110001);
                                __m256 max256   = _mm256_max_ps(src256, src256_2);

                                __m128 src128_0 = _mm256_extractf128_ps(max256, 0);
                                __m128 src128_1 = _mm256_extractf128_ps(max256, 1);

                                __m128 src128   = _mm_shuffle_ps(src128_0, src128_1, 0b10001000);

                                max128 = _mm_max_ps(src128, max128);

                            }
                        }

                        _mm_storeu_ps(&dst[outIndex], max128);
                    }
                }

                for(; j<outWidth; ++j)       

                {
                    int outIndex  =  j + outWidth * (i + outHeight * (k + channel * b));
                    float max     =  -FLT_MAX;
                    int maxIndex  =  -1;

                    for(int n=0; n<kSizeY; ++n)
                    {
                        for(int m=0; m<kSizeX; ++m)
                        {
                            int curHeight    = heightOffset + i*stride + n;
                            int curWidth     = widthOffset  + j*stride + m;

                            int index        = curWidth + width*(curHeight + height*(k + b*channel));
                            int valid        = (curHeight >=0 && curHeight < height && curWidth>=0 && curWidth<width );

                            float value      = (valid !=0)?src[index]: -FLT_MAX;

                            maxIndex         = (value > max)?index:maxIndex;
                            max              = (value > max)?value:max;
                        }
                    }

                    dst[outIndex]  = max;
                }
            }
        }
    }
}
#endif

MaxPoolLayer::~MaxPoolLayer()
{
#ifdef USE_GPU
#ifdef USE_CUDNN
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_outputDesc));
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(_maxPoolDesc));
#endif
#endif
}

int MaxPoolLayer::getKSizeX() const
{
    return _kSizeX;
}

int MaxPoolLayer::getKSizeY() const
{
    return _kSizeY;
}

int MaxPoolLayer::getStride() const
{
    return _stride;
}

int MaxPoolLayer::getStrideX() const
{
    return _strideX;
}

int MaxPoolLayer::getStrideY() const
{
    return _strideY;
}

int MaxPoolLayer::getPaddingX() const
{
    return _paddingX;
}

int MaxPoolLayer::getPaddingY() const
{
    return _paddingY;
}

int MaxPoolLayer::getAntialiasing() const
{
    return _antialiasing;
}

int MaxPoolLayer::getMaxPoolDepth() const
{
    return _maxPoolDepth;
}

int MaxPoolLayer::getOutChannelsMp() const
{
    return _outChannelsMp;
}

int MaxPoolLayer::getCeilMode() const
{
    return _ceilMode;
}

}
