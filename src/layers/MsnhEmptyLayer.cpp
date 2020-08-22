#include "Msnhnet/layers/MsnhEmptyLayer.h"
namespace Msnhnet
{
EmptyLayer::EmptyLayer(const int &batch, const int &width, const int &height, const int &channel)
{
    this->_layerName =  "Empty           ";
    this->_type      =   LayerType::EMPTY;
    this->_batch     =   batch;
    this->_width     =   width;
    this->_height    =   height;
    this->_channel   =   channel;

    this->_outWidth  =   width;
    this->_outHeight =   height;
    this->_outChannel=   channel;
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
    sprintf_s(msg, "Empty Layer                  %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
              this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#else
    sprintf(msg, "Empty Layer                  %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
            this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#endif
    this->_layerDetail = msg;
}

EmptyLayer::~EmptyLayer()
{
}

void EmptyLayer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();
    Blas::cpuCopy(netState.inputNum*this->_batch, netState.input, 1, this->_output, 1);
    this->_forwardTime = TimeUtil::getElapsedTime(st);
}

#ifdef USE_GPU
void EmptyLayer::forwardGPU(NetworkState &netState)
{

    this->recordCudaStart();
    BlasGPU::gpuSimpleCopy(netState.inputNum, netState.input, this->_gpuOutput);
    this->recordCudaStop();
}
#endif
}
