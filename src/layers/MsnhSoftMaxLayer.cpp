#include "Msnhnet/layers/MsnhSoftMaxLayer.h"

namespace Msnhnet
{
SoftMaxLayer::SoftMaxLayer(const int &batch, const int &inputNum, const int &groups)
{
    this->type          =   LayerType::SOFTMAX;
    this->layerName     =   "SoftMax         ";

    this->batch         =   batch;
    this->groups        =   groups;
    this->inputNum      =   inputNum;
    this->outputNum     =   outputNum;

    if(!BaseLayer::isPreviewMode)
    {
        this->output        =   new float[static_cast<size_t>(this->inputNum * this->batch)]();
    }

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "softmax                                        %4d\n", inputNum);
#else
    sprintf(msg, "softmax                                        %4d\n", inputNum);
#endif
    this->layerDetail   = msg;
}

void SoftMaxLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    Blas::cpuSoftmax(netState.input, this->inputNum/this->groups, this->batch, this->inputNum,
                     this->groups, this->inputNum/this->groups, this->temperature, 1, this->output);

    auto so = std::chrono::system_clock::now();
    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
}
}
