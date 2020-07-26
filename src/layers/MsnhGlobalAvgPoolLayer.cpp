#include "Msnhnet/layers/MsnhGlobalAvgPoolLayer.h"
namespace Msnhnet
{
GlobalAvgPoolLayer::GlobalAvgPoolLayer(const int &batch, const int &height, const int &width, const int &channel)
{
    this->_type              = LayerType::GLOBAL_AVGPOOL;

    this->_layerName         = "GlobalAvgPool   ";

    this->_batch             = batch;
    this->_height            = height;
    this->_width             = width;
    this->_channel           = channel;

    this->_outWidth          = 1;
    this->_outHeight         = 1;
    this->_outChannel        = channel;
    this->_inputNum          = height*width*channel;
    this->_outputNum         = this->_outChannel;

    this->_output            = new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
    if(!BaseLayer::isPreviewMode)
    {
        this->_bFlops            = (this->_width*this->_height* this->_channel*this->_outHeight*this->_outWidth)/ 1000000000.f;
    }

    char msg[100];

#ifdef WIN32
        sprintf_s(msg, "GlobalAvgPool                %4d x%4d x%4d ->   %4d\n %5.3f BF\n",
                  this->_width, this->_height, this->_channel, this->_outChannel, this->_bFlops);
#else
        sprintf(msg, "GlobalAvgPool                %4d x%4d x%4d ->   %4d\n %5.3f BF\n",
              this->_width, this->_height, this->_channel, this->_outChannel, this->_bFlops);
#endif
}

GlobalAvgPoolLayer::~GlobalAvgPoolLayer()
{

}

void GlobalAvgPoolLayer::forward(NetworkState &netState)
{

    for (int b = 0; b < this->_batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int k = 0; k < this->_channel; ++k)
        {
            int outIndex = k + b*this->_channel;
            this->_output[outIndex] = 0;
            for (int i = 0; i < this->_height*this->_width; ++i)
            {
                int inIndex = i + this->_height*this->_width*(k + b*this->_channel);
                this->_output[outIndex] += netState.input[inIndex];
            }
            this->_output[outIndex] /= (this->_height*this->_width);
        }
    }
}

}

