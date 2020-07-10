#include "Msnhnet/layers/MsnhPaddingLayer.h"
namespace Msnhnet
{

PaddingLayer::PaddingLayer(const int &batch, const int &height, const int &width, const int &channel, const int &top,
                           const int &down, const int &left, const int &right, const float &paddingVal)
{
    this->type          =   LayerType::PADDING;
    this->layerName     =   "Padding         ";
    this->batch         =   batch;
    this->height        =   height;
    this->width         =   width;
    this->channel       =   channel;

    this->inputNum      =   width * height * channel;

    this->top           =   top;
    this->down          =   down;
    this->left          =   left;
    this->right         =   right;

    this->paddingVal    =   paddingVal;

    this->outHeight     =   this->height + this->top + this->down;
    this->outWidth      =   this->width   + this->left + this->right;
    this->outChannel    =   this->channel;

    this->outputNum     =   this->outHeight * this->outWidth * this->outChannel;

    if(!BaseLayer::isPreviewMode)
    {
        this->output        =   new float[static_cast<size_t>(this->outputNum * this->batch)]();
    }

    char msg[100];

#ifdef WIN32
    sprintf_s(msg, "padding                      %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->width, this->height, this->channel,
              this->outWidth, this->outHeight, this->outChannel, this->bFlops);
#else
    sprintf(msg, "padding                      %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->width, this->height, this->channel,
            this->outWidth, this->outHeight, this->outChannel, this->bFlops);
#endif
    this->layerDetail = msg;
}

void PaddingLayer::forward(NetworkState &netState)
{
    for (int i = 0; i < this->batch; ++i)
    {
        for (int j = 0; j < this->outChannel; ++j)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int m = 0; m < this->outHeight; ++m)
            {
                for (int n = 0; n < this->outWidth; ++n)
                {
                    float val = 0;

                    if(m < this->top || (m >= (this->height + this->top)))
                    {
                        val     =   this->paddingVal;
                    }
                    else
                    {
                        if(n < this->left || (n >= (this->width + this->left)))
                        {
                            val     =   this->paddingVal;
                        }
                        else
                        {
                            val     =   netState.input[ i*this->channel*this->height*this->width + j*this->height*this->width + (m-this->top)*this->width + (n - this->left)];
                        }
                    }

                    this->output[i*this->outChannel*this->outHeight*this->outHeight + j*this->outHeight*this->outWidth + m*this->outWidth + n] = val;

                }
            }
        }
    }

}

}
