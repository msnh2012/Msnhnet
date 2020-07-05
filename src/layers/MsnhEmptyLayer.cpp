#include "Msnhnet/layers/MsnhEmptyLayer.h"
namespace Msnhnet
{
EmptyLayer::EmptyLayer(const int &batch, const int &width, const int &height, const int &channel)
{
    this->layerName =  "Empty           ";
    this->type      =   LayerType::EMPTY;
    this->batch     =   batch;
    this->width     =   width;
    this->height    =   height;
    this->channel   =   channel;

    this->outWidth  =   width;
    this->outHeight =   height;
    this->outChannel=   channel;
    this->inputNum  =   width * height * channel;
    this->outputNum =   this->outWidth * this->outHeight * this->outChannel;

    if(!BaseLayer::isPreviewMode)
    {
        this->output    =   new float[static_cast<size_t>(this->outputNum * this->batch)]();
    }
    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Empty Layer                  %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->width, this->height, this->channel,
              this->outWidth, this->outHeight, this->outChannel, this->bFlops);
#else
    sprintf(msg, "Empty Layer                  %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->width, this->height, this->channel,
            this->outWidth, this->outHeight, this->outChannel, this->bFlops);
#endif
    this->layerDetail = msg;
}

EmptyLayer::~EmptyLayer()
{
}

void EmptyLayer::forward(NetworkState &netState)
{
    Blas::cpuCopy(netState.inputNum, netState.input, 1, this->output, 1);
}
}
