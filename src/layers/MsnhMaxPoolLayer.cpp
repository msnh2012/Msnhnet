#include "Msnhnet/layers/MsnhMaxPoolLayer.h"

namespace Msnhnet
{
MaxPoolLayer::MaxPoolLayer(const int &batch,   const int &height, const int &width, const int &channel, const int &kSizeX,  const int &kSizeY,
                           const int &strideX, const int &strideY, const int &paddingX, const int &paddingY, const int &maxPoolDepth,
                           const int &outChannelsMp, const int& ceilMode,  const int &antialiasing)
{
    this->type           = LayerType::MAXPOOL;
    this->layerName      = "MaxPool         ";

    this->batch          = batch;
    this->height         = height;
    this->width          = width;
    this->channel        = channel;

    this->paddingX       = paddingX;
    this->paddingY       = paddingY;

    this->ceilMode       = ceilMode;

    this->maxPoolDepth   = maxPoolDepth;
    this->outChannelsMp  = outChannelsMp;   
    this->antialiasing   = antialiasing;

    this->kSizeX         = kSizeX;  
    this->kSizeY         = kSizeY;  

    this->stride         = strideX; 
    this->strideX        = strideX;
    this->strideY        = strideY;

    if(maxPoolDepth)
    {
        this->outChannel = outChannelsMp;
        this->outWidth   = this->width;
        this->outHeight  = this->height;
    }
    else
    {

        if(this->ceilMode == 1)
        {
            int tmpW = (width  + paddingX*2 - kSizeX) % strideX; 
            int tmpH = (height + paddingY*2 - kSizeY) % strideY; 

            if(tmpW >= kSizeX)
            {
                throw Exception(1,"maxpool padding error ", __FILE__, __LINE__);
            }

            if(tmpH >= kSizeY)
            {
                throw Exception(1,"maxpool padding error ", __FILE__, __LINE__);
            }

            if(tmpW <= paddingX)
            {
                this->outWidth   = (width  + paddingX*2 - kSizeX) / strideX + 1; 
            }
            else
            {
                this->outWidth   = (width  + paddingX*2 - kSizeX) / strideX + 2; 
            }

            if(tmpH <= paddingY)
            {
                this->outHeight  = (height + paddingY*2 - kSizeY) / strideY + 1; 
            }
            else
            {
                this->outHeight  = (height + paddingY*2 - kSizeY) / strideY + 2; 
            }
        }
        else if(this->ceilMode == 0)
        {
            this->outWidth   = (width  + paddingX*2 - kSizeX) / strideX + 1; 
            this->outHeight  = (height + paddingY*2 - kSizeY) / strideY + 1; 
        }
        else
        {
            this->outWidth   = (width  + paddingX - kSizeX) / strideX + 1; 
            this->outHeight  = (height + paddingY - kSizeY) / strideY + 1; 
        }
        this->outChannel = channel;                                 
    }

    this->outputNum      =  this->outHeight * this->outWidth * this->outChannel; 
    this->inputNum       =  height * width * channel; 

    if(!BaseLayer::isPreviewMode)
    {

        this->output         = new float[static_cast<size_t>(this->outputNum * this->batch)]();
    }

    this->bFlops            = (this->kSizeX*this->kSizeY* this->channel*this->outHeight*this->outWidth)/ 1000000000.f;

    char msg[100];

    if(maxPoolDepth)
    {
#ifdef WIN32
        sprintf_s(msg, "max-depth         %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  this->kSizeX, this->kSizeY, this->strideX, this->width, this->height, this->channel,
                  this->outWidth, this->outHeight,this->outChannel,this->bFlops);
#else
        sprintf(msg, "max-depth         %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                this->kSizeX, this->kSizeY, this->strideX, this->width, this->height, this->channel,
                this->outWidth, this->outHeight,this->outChannel,this->bFlops);
#endif
    }

    if(strideX == strideY)
    {
#ifdef WIN32
        sprintf_s(msg, "max               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  this->kSizeX, this->kSizeY, this->strideX, this->width, this->height, this->channel,
                  this->outWidth, this->outHeight,this->outChannel,this->bFlops);
#else
        sprintf(msg, "max               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                this->kSizeX, this->kSizeY, this->strideX, this->width, this->height, this->channel,
                this->outWidth, this->outHeight,this->outChannel,this->bFlops);
#endif
    }
    else
    {
#ifdef WIN32
        sprintf_s(msg, "max              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  this->kSizeX, this->kSizeY, this->strideX,this->strideY, this->width, this->height,
                  this->channel, this->outWidth, this->outHeight,this->outChannel,this->bFlops);
#else
        sprintf(msg, "max              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                this->kSizeX, this->kSizeY, this->strideX,this->strideY, this->width, this->height,
                this->channel, this->outWidth, this->outHeight,this->outChannel,this->bFlops);
#endif
    }

    this->layerDetail       = msg;

}

void MaxPoolLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    if(this->maxPoolDepth)
    {
        for(int b=0; b<this->batch; ++b)                    
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif                                                      
            for(int i=0; i<this->height; i++)               
            {
                for(int j=0; j<this->width; ++j)            
                {
                    for(int g=0; j<this->outChannel; ++g)   
                    {
                        int outIndex = j + this->width*(i + this->height*(g + this->outChannel*b));
                        float max    = -FLT_MAX;
                        int maxIndex = -1;

                        for(int k=g; k<this->channel; k+=this->outChannel)
                        {
                            int inIndex = j + this->width*(i + this->height*(k + this->channel*b));
                            float val   = netState.input[inIndex];

                            maxIndex    = (val > max)? inIndex:maxIndex;
                            max         = (val > max)? val:max;
                        }

                        this->output[outIndex] = max;
                    }

                }

            }
        }
        return;
    }
#ifdef USE_X86
    if((this->strideX == this->strideY) && supportAvx )
    {
        forwardAvx(netState.input,this->output,this->kSizeX, this->kSizeY, this->width,this->height,this->outWidth,
                   this->outHeight,this->channel,this->paddingX, this->paddingY,this->stride,this->batch);
    }
    else
#endif
    {

        int widthOffset  =     -(this->paddingX + 1)/2;
        int heightOffset =     -(this->paddingY + 1)/2;

        int mHeight         =   this->outHeight;
        int mWidth          =   this->outWidth;

        int mChannel        =   this->channel;

        for(int b=0; b<this->batch; ++b)                
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

                        int outIndex = j + mWidth*(i + mHeight*(k + channel*b));
                        float max    = -FLT_MAX;
                        int maxIndex = -1;

                        for(int n=0; n<this->kSizeY; ++n)
                        {
                            for(int m=0; m<this->kSizeX; ++m)
                            {

                                int curHeight =  heightOffset + i*this->strideY + n;

                                int curWidth  =  widthOffset  + j*this->strideX + m;

                                int index     =  curWidth + this->width*(curHeight + this->height*(k + b*this->channel));

                                bool valid    =  (curHeight >=0 && curHeight < this->height &&
                                                  curWidth  >=0 && curWidth  < this->width);

                                float value   =  (valid)? netState.input[index] : -FLT_MAX;

                                maxIndex      =  (value > max) ? index : maxIndex;

                                max           =  (value > max) ? value : max;
                            }
                        }

                        this->output[outIndex] = max;

                    }
                }
            }
        }
    }

    auto so = std::chrono::system_clock::now();
    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

#ifdef USE_X86
void MaxPoolLayer::forwardAvx(float *const &src, float *const &dst, const int &kSizeX, const int &kSizeY,
                              const int &width, const int &height, const int &outWidth, const int &outHeight,
                              const int &channel,const int &paddingX,const int &paddingY,const int &stride, const int &batch)
{

    int widthOffset  =     -(paddingX + 1)/2;
    int heightOffset =     -(paddingY + 1)/2;

    for(int b=0; b<batch; ++b)
    {
#pragma omp parallel for num_threads(OMP_THREAD)
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

}

}
