#include "Msnhnet/layers/MsnhSoftMaxLayer.h"

namespace Msnhnet
{
SoftMaxLayer::SoftMaxLayer(const int &batch, const int &inputNum, const int &groups, const float &temperature)
{
    this->_type          =   LayerType::SOFTMAX;
    this->_layerName     =   "SoftMax         ";

    this->_batch         =   batch;
    this->_groups        =   groups;
    this->_temperature   =   temperature;
    this->_inputNum      =   inputNum;
    this->_outputNum     =   inputNum;

    if(!BaseLayer::isPreviewMode)
    {
        this->_output        =   new float[static_cast<size_t>(this->_inputNum * this->_batch)]();
    }

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "softmax                                        %4d\n", inputNum);
#else
    sprintf(msg, "softmax                                        %4d\n", inputNum);
#endif
    this->_layerDetail   = msg;
}

void SoftMaxLayer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    Blas::cpuSoftmax(netState.input, this->_inputNum/this->_groups, this->_batch, this->_inputNum,
                     this->_groups, this->_inputNum/this->_groups, this->_temperature, 1, this->_output, BaseLayer::supportAvx);

    auto so = std::chrono::system_clock::now();
    this->_forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
}

int SoftMaxLayer::getGroups() const
{
    return _groups;
}

float SoftMaxLayer::getTemperature() const
{
    return _temperature;
}
}
