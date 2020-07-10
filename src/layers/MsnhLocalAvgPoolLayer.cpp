#include "Msnhnet/layers/MsnhLocalAvgPoolLayer.h"

namespace Msnhnet
{
LocalAvgPoolLayer::LocalAvgPoolLayer(const int &batch, const int &height, const int &width, const int &channel,
                                     const int &kSizeX, const int &kSizeY, const int &strideX, const int &strideY, const int &paddingX, const int &paddingY, const int &ceilMode,
                                     const int &antialiasing)
{

    this->type              = LayerType::LOCAL_AVGPOOL;
    this->layerName         = "LocalAvgPool    ";

    this->batch             = batch;
    this->height            = height;
    this->width             = width;
    this->channel           = channel;
    this->paddingX          = paddingX;
    this->paddingY          = paddingY;

    this->kSizeX            = kSizeX;

    this->kSizeY            = kSizeY;

    this->ceilMode          = ceilMode;

    this->antialiasing      = antialiasing;

    if(antialiasing)
    {
        this->strideX       = 1;
        this->strideY       = 1;
    }
    else
    {
        this->strideX       = strideX;

        this->strideY       = strideY;

    }

    if(this->ceilMode == 1)
    {
        int tmpW = (width  + paddingX*2 - kSizeX) % strideX;

        int tmpH = (height + paddingY*2 - kSizeY) % strideY;

        if(tmpW >= kSizeX)
        {
            throw Exception(1,"localavgpool padding error ", __FILE__, __LINE__);
        }

        if(tmpH >= kSizeY)
        {
            throw Exception(1,"localavgpool padding error ", __FILE__, __LINE__);
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
        this->outWidth   = (width  + 2*paddingX - kSizeX) / strideX + 1;

        this->outHeight  = (height + 2*paddingY - kSizeY) / strideY + 1;

    }
    else
    {
        this->outWidth   = (width  + paddingX - kSizeX) / strideX + 1;

        this->outHeight  = (height + paddingY - kSizeY) / strideY + 1;

    }

    this->outChannel        = channel;

    this->outputNum         = this->outHeight * this->outWidth * this->outChannel;

    this->inputNum          = height * width * channel;

    if(!BaseLayer::isPreviewMode)
    {
        this->output            = new float[static_cast<size_t>(outputNum * this->batch)]();
    }

    this->bFlops            = (this->kSizeX*this->kSizeY* this->channel*this->outHeight*this->outWidth)/ 1000000000.f;

    char msg[100];

    if(strideX == strideY)
    {
#ifdef WIN32
        sprintf_s(msg, "avg               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  this->kSizeX, this->kSizeY, this->strideX, this->width, this->height, this->channel,
                  this->outWidth, this->outHeight,this->outChannel,this->bFlops);
#else
        sprintf(msg, "avg               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                this->kSizeX, this->kSizeY, this->strideX, this->width, this->height, this->channel,
                this->outWidth, this->outHeight,this->outChannel,this->bFlops);
#endif
    }
    else
    {
#ifdef WIN32
        sprintf_s(msg, "avg              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                  this->kSizeX, this->kSizeY, this->strideX,this->strideY, this->width, this->height,
                  this->channel, this->outWidth, this->outHeight,this->outChannel,this->bFlops);
#else
        sprintf(msg, "avg              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n",
                this->kSizeX, this->kSizeY, this->strideX,this->strideY, this->width, this->height,
                this->channel, this->outWidth, this->outHeight,this->outChannel,this->bFlops);
#endif
    }

    this->layerDetail       = msg;

}

LocalAvgPoolLayer::~LocalAvgPoolLayer()
{

}

void LocalAvgPoolLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    int widthOffset  =     -(this->paddingX + 1) / 2;
    int heightOffset =     -(this->paddingY + 1) / 2;

    int mHeight         =   this->outHeight;
    int mWidth          =   this->outWidth;

    int mChannel        =   this->channel;

    for(int b=0; b<this->batch; ++b)

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

                    int outIndex = j + mWidth*(i + mHeight*(k + channel*b));

                    float avg    = 0;

                    int counter  = 0;

                    for(int n=0; n<this->kSizeY; ++n)
                    {
                        for(int m=0; m<this->kSizeX; ++m)
                        {

                            int curHeight =  heightOffset + i*this->strideY + n;

                            int curWidth  =  widthOffset  + j*this->strideX + m;

                            int index     =  curWidth + this->width*(curHeight + this->height*(k + b*this->channel));

                            bool valid    =  (curHeight >=0 && curHeight < this->height &&
                                              curWidth  >=0 && curWidth  < this->width);

                            if(valid)
                            {
                                counter++;
                                avg += netState.input[index];
                            }
                        }
                    }

                    this->output[outIndex] = avg / counter;

                }

            }
        }
    }

    auto so = std::chrono::system_clock::now();
    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

}
