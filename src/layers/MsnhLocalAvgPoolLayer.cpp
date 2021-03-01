#include "Msnhnet/layers/MsnhLocalAvgPoolLayer.h"

namespace Msnhnet
{
LocalAvgPoolLayer::LocalAvgPoolLayer(const int &batch, const int &height, const int &width, const int &channel,
                                     const int &kSizeX, const int &kSizeY, const int &strideX, const int &strideY, const int &paddingX, const int &paddingY, const int &ceilMode,
                                     const int &antialiasing)
{

    this->_type              = LayerType::LOCAL_AVGPOOL;
    this->_layerName         = "LocalAvgPool    ";

    this->_batch             = batch;
    this->_height            = height;
    this->_width             = width;
    this->_channel           = channel;
    this->_paddingX          = paddingX;
    this->_paddingY          = paddingY;

    this->_kSizeX            = kSizeX;  

    this->_kSizeY            = kSizeY;

    this->_ceilMode          = ceilMode;

    this->_antialiasing      = antialiasing;

    if(antialiasing)
    {
        this->_strideX       = 1;
        this->_strideY       = 1;
    }
    else
    {
        this->_strideX       = strideX;

        this->_strideY       = strideY;

    }

    if(this->_ceilMode == 1)
    {
        int tmpW = (width  + paddingX*2 - kSizeX) % strideX; 

        int tmpH = (height + paddingY*2 - kSizeY) % strideY; 

        if(tmpW >= kSizeX)
        {
            throw Exception(1,"localavgpool padding error ", __FILE__, __LINE__, __FUNCTION__);
        }

        if(tmpH >= kSizeY)
        {
            throw Exception(1,"localavgpool padding error ", __FILE__, __LINE__, __FUNCTION__);
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
        this->_outWidth   = (width  + 2*paddingX - kSizeX) / strideX + 1; 

        this->_outHeight  = (height + 2*paddingY - kSizeY) / strideY + 1; 

    }
    else
    {
        this->_outWidth   = (width  + paddingX - kSizeX) / strideX + 1; 

        this->_outHeight  = (height + paddingY - kSizeY) / strideY + 1; 

    }

    this->_outChannel        = channel;                                  

    this->_outputNum         = this->_outHeight * this->_outWidth * this->_outChannel; 

    this->_inputNum          = height * width * channel; 

#ifdef USE_GPU
#ifdef USE_CUDNN

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->_inputDesc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->_inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->_batch, this->_channel, this->_height, this->_width));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&this->_outputDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(this->_outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->_batch, this->_outChannel, this->_outHeight, this->_outWidth));

    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&this->_localAvgPoolDesc));

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(this->_localAvgPoolDesc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN,  

                                            this->_kSizeY, this->_kSizeX, this->_paddingY, this->_paddingX, this->_strideY,this->_strideX));
#endif
#endif

    this->_bFlops            = (this->_kSizeX*this->_kSizeY* this->_channel*this->_outHeight*this->_outWidth)/ 1000000000.f;

    this->_maxOutputNum  = this->_batch*this->_outputNum;

    char msg[100];

    if(strideX == strideY)
    {
#ifdef WIN32
        sprintf_s(msg, "LocalAvgPool       %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  this->_kSizeX, this->_kSizeY, this->_strideX, this->_width, this->_height, this->_channel,
                  this->_outWidth, this->_outHeight,this->_outChannel,this->_bFlops);
#else
        sprintf(msg, "avg               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                this->_kSizeX, this->_kSizeY, this->_strideX, this->_width, this->_height, this->_channel,
                this->_outWidth, this->_outHeight,this->_outChannel,this->_bFlops);
#endif
    }
    else
    {
#ifdef WIN32
        sprintf_s(msg, "LocalAvgPool       %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  this->_kSizeX, this->_kSizeY, this->_strideX,this->_strideY, this->_width, this->_height,
                  this->_channel, this->_outWidth, this->_outHeight,this->_outChannel,this->_bFlops);
#else
        sprintf(msg, "avg              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                this->_kSizeX, this->_kSizeY, this->_strideX,this->_strideY, this->_width, this->_height,
                this->_channel, this->_outWidth, this->_outHeight,this->_outChannel,this->_bFlops);
#endif
    }

    this->_layerDetail       = msg;

}

LocalAvgPoolLayer::~LocalAvgPoolLayer()
{
#ifdef USE_GPU
#ifdef USE_CUDNN
    if(_inputDesc)
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(_inputDesc));

    if(_outputDesc)
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(_outputDesc));

    if(_localAvgPoolDesc)
        CUDNN_CHECK(cudnnDestroyPoolingDescriptor(_localAvgPoolDesc));
#endif
#endif
}

int LocalAvgPoolLayer::getKSizeX() const
{
    return _kSizeX;
}

int LocalAvgPoolLayer::getKSizeY() const
{
    return _kSizeY;
}

int LocalAvgPoolLayer::getStride() const
{
    return _stride;
}

int LocalAvgPoolLayer::getStrideX() const
{
    return _strideX;
}

int LocalAvgPoolLayer::getStrideY() const
{
    return _strideY;
}

int LocalAvgPoolLayer::getPaddingX() const
{
    return _paddingX;
}

int LocalAvgPoolLayer::getPaddingY() const
{
    return _paddingY;
}

int LocalAvgPoolLayer::getCeilMode() const
{
    return _ceilMode;
}

int LocalAvgPoolLayer::getAntialiasing() const
{
    return _antialiasing;
}

void LocalAvgPoolLayer::forward(NetworkState &netState)
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

    int widthOffset  =     -this->_paddingX;
    int heightOffset =     -this->_paddingY;

    int mHeight         =   this->_outHeight;
    int mWidth          =   this->_outWidth;

    int mChannel        =   this->_channel;

    for(int b=0; b<this->_batch; ++b)                

    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int k=0; k<mChannel; ++k)               

        {
            for(int i=0; i<mHeight; ++i)            

            {
                for(int j=0; j<mWidth; ++j)         

                {

                    int outIndex = j + mWidth*(i + mHeight*(k + _channel*b));

                    float avg    = 0;

                    int counter  = 0;

                    for(int n=0; n<this->_kSizeY; ++n)
                    {
                        for(int m=0; m<this->_kSizeX; ++m)
                        {

                            int curHeight =  heightOffset + i*this->_strideY + n;

                            int curWidth  =  widthOffset  + j*this->_strideX + m;

                            int index     =  curWidth + this->_width*(curHeight + this->_height*(k + b*this->_channel));

                            bool valid    =  (curHeight >=0 && curHeight < this->_height &&
                                              curWidth  >=0 && curWidth  < this->_width);

                            if(valid)
                            {
                                counter++;
                                avg += layerInput[index];
                            }
                        }
                    }

                    layerOutput[outIndex] = avg / counter;  

                }

            }
        }
    }

    this->_forwardTime =   TimeUtil::getElapsedTime(st);

}

void LocalAvgPoolLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu) 

            {

                this->_output             = MemoryManager::effcientNew<float>(static_cast<size_t>(this->_outputNum * this->_batch));
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
void LocalAvgPoolLayer::forwardGPU(NetworkState &netState)
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

#ifdef USE_CUDNN
    if(!onlyUseCuda)
    {
        float a = 1.f;
        float b = 0;
        CUDNN_CHECK(cudnnPoolingForward(Cuda::getCudnnHandle(), this->_localAvgPoolDesc, &a,
                                        this->_inputDesc, layerGpuInput,
                                        &b,
                                        this->_outputDesc, layerGpuOutput));
    }
    else
    {
        LocalAvgPoolLayerGPU::forwardNormalGPU(this->_width,this->_height,this->_channel,
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
    LocalAvgPoolLayerGPU::forwardNormalGPU(this->_width,this->_height,this->_channel,
                                           this->_outWidth, this->_outHeight, this->_outChannel,
                                           this->_strideX, this->_strideY,
                                           this->_kSizeX, this->_kSizeY,
                                           this->_paddingX, this->_paddingY,
                                           this->_batch,
                                           layerGpuInput,
                                           layerGpuOutput
                                           );
#endif
    this->recordCudaStop();
}
#endif

}
