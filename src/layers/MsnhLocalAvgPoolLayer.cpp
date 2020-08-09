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

    if(!BaseLayer::isPreviewMode)
    {
        this->_output         = new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
#ifdef USE_GPU
        this->_gpuOutput      = Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
#endif
    }

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
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_inputDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(_outputDesc));
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

    TimeUtil::startRecord();

    int widthOffset  =     -(this->_paddingX + 1) / 2;
    int heightOffset =     -(this->_paddingY + 1) / 2;

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
                                avg += netState.input[index];
                            }
                        }
                    }

                    this->_output[outIndex] = avg / counter;  

                }

            }
        }
    }

    this->_forwardTime =   TimeUtil::getElapsedTime();

}

#ifdef USE_GPU
void LocalAvgPoolLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();
#ifdef USE_CUDNN
    if(!onlyUseCuda)
    {
        float a = 1.f;
        float b = 0;
        CUDNN_CHECK(cudnnPoolingForward(Cuda::getCudnnHandle(), this->_localAvgPoolDesc, &a,
                                        this->_inputDesc, netState.input,
                                        &b,
                                        this->_outputDesc, this->_gpuOutput));
    }
    else
    {
        LocalAvgPoolLayerGPU::forwardNormalGPU(this->_width,this->_height,this->_channel,
                                               this->_outWidth, this->_outHeight, this->_outChannel,
                                               this->_strideX, this->_strideY,
                                               this->_kSizeX, this->_kSizeY,
                                               this->_paddingX, this->_paddingY,
                                               this->_batch,
                                               netState.input,
                                               this->_gpuOutput
                                               );
    }
#else
    LocalAvgPoolLayerGPU::forwardNormalGPU(this->_width,this->_height,this->_channel,
                                           this->_outWidth, this->_outHeight, this->_outChannel,
                                           this->_strideX, this->_strideY,
                                           this->_kSizeX, this->_kSizeY,
                                           this->_paddingX, this->_paddingY,
                                           this->_batch,
                                           netState.input,
                                           this->_gpuOutput
                                           );
#endif
    this->recordCudaStop();
}
#endif

}
