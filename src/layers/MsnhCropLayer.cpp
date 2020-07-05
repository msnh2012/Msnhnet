#include "Msnhnet/layers/MsnhCropLayer.h"

namespace Msnhnet
{
CropLayer::CropLayer(const int &batch, const int &height, const int &width, const int &channel, const int &cropHeight, const int &cropWidth, const int &flip, const float &angle, const float &saturation, const float &exposure)
{
    (void)angle;
    this->type          =   LayerType::CROP;
    this->layerName     =  "Crop            ";

   this->batch         =   batch;
    this->height        =   height;
    this->width         =   width;
    this->channel       =   channel;
    this->scale         =   static_cast<float>(cropHeight/height);
    this->flip          =   flip;
    this->saturation    =   saturation;
    this->exposure      =   exposure;
    this->outWidth      =   cropWidth;
    this->outHeight     =   cropHeight;
    this->outChannel    =   channel;

   this->outputNum     =   this->outWidth * this->outHeight * this->outChannel;
    this->inputNum      =   width * height * channel;

   if(!BaseLayer::isPreviewMode)
    {
        this->output        =   new float[static_cast<size_t>(this->outputNum * this->batch)]();
    }

   char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Crop Layer: %d x %d -> %d x %d x %d image\n", height, width, cropHeight, cropWidth, channel);
#else
    sprintf(msg, "Crop Layer: %d x %d -> %d x %d x %d image\n", height, width, cropHeight, cropWidth, channel);
#endif
    this->layerDetail   = msg;
}

void CropLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

   int flip        =   (this->flip && rand() % 2);
    int dh          =   rand()%(this->height - this->outHeight + 1);
    int dw          =   rand()%(this->width  - this->outWidth  + 1);
    float scale     =   2;
    float trans     =  -1;
    int   col       =   0;
    int   row       =   0;
    int   index     =   0;
    int   cnt       =   0;

   if(this->noAdjust)
    {
        scale       =   1;
        trans       =   0;
    }

   flip        =   0;
    dw          =   (this->height - this->outHeight)/2;
    dh          =   (this->width  - this->outWidth)/2;

   for (int b = 0; b < this->batch; ++b)
    {
        for (int c = 0; c < this->channel; ++c)
        {
            for (int i = 0; i < this->outHeight; ++i)
            {
                for (int j = 0; j < this->outWidth; ++j)
                {
                    if(flip)
                    {
                        col = this->width - dw - j -1;
                    }
                    else
                    {
                        col = j + dw;
                    }

                   row     = i + dh;
                    index   = col + this->width *(row + this->height*(c + this->channel*b));
                    this->output[cnt++]   =   netState.input[index]*scale + trans;
                }
            }
        }
    }

   auto so = std::chrono::system_clock::now();

   this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

void CropLayer::resize(const int &width, const int &height)
{
    this->width     =   width;
    this->height    =   height;

   this->outHeight =   outHeight;
    this->outWidth  =   outWidth;

   this->inputNum  =   this->width * this->height * this->channel;
    this->outputNum =   this->outHeight * this->outWidth * this->outChannel;

   if(this->output == nullptr)
    {
        throw Exception(1,"output can't be null", __FILE__, __LINE__);
    }

   this->output    = static_cast<float*>(realloc(this->output, static_cast<size_t>(this->batch*this->outputNum)*sizeof(float)));
}
}
