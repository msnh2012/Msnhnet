#include "Msnhnet/layers/MsnhViewLayer.h"
namespace Msnhnet
{

ViewLayer::ViewLayer(const int &batch, const int &width, const int &height, const int &channel, const int &outWidth, const int &outHeight, const int &outChannel)
{
    if((width*height*channel)!=(outWidth*outHeight*outChannel))
    {
        throw Exception(1,"view op params error",__FILE__,__LINE__,__FUNCTION__);
    }

    this->_layerName =  "View            ";
    this->_type      =   LayerType::VIEW;
    this->_batch     =   batch;
    this->_width     =   width;
    this->_height    =   height;
    this->_channel   =   channel;

    this->_outWidth  =   outWidth;
    this->_outHeight =   outHeight;
    this->_outChannel=   outChannel;
    this->_inputNum  =   width * height * channel;
    this->_outputNum =   this->_outWidth * this->_outHeight * this->_outChannel;

    if(!BaseLayer::isPreviewMode)
    {
        this->_output    =   new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
#ifdef USE_GPU
        this->_gpuOutput         = Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
#endif
    }
    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "View                         %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
              this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#else
    sprintf(msg, "View Layer                   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
            this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#endif
    this->_layerDetail = msg;
}

ViewLayer::~ViewLayer()
{

}

void ViewLayer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();
    Blas::cpuCopy(netState.inputNum*this->_batch, netState.input, 1, this->_output, 1);
    this->_forwardTime = TimeUtil::getElapsedTime(st);
}

#ifdef USE_GPU
void ViewLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();
    BlasGPU::gpuSimpleCopy(netState.inputNum, netState.input, this->_gpuOutput);
    this->recordCudaStop();
}
#endif

}

