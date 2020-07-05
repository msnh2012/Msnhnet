#include "Msnhnet/layers/MsnhActivationLayer.h"

namespace Msnhnet
{
ActivationLayer::ActivationLayer(const int &batch, const int &inputNum, const ActivationType &activation)
{
    this->type          = LayerType::ACTIVE;
    this->inputNum      = inputNum;
    this->outputNum     = outputNum;
    this->batch         = batch;
    this->activation    = activation;

    if(!BaseLayer::isPreviewMode)
    {
        this->output        = new float[static_cast<size_t>(batch*outputNum)]();
    }
}

void ActivationLayer::forward(NetworkState &netState)
{
    Blas::cpuCopy(this->outputNum*this->batch,
                  netState.input,
                  1,
                  output,
                  1);

    Activations::activateArray(output,
                               outputNum*batch,
                               activation);
}
}
