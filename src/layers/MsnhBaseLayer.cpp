#include "Msnhnet/layers/MsnhBaseLayer.h"

namespace Msnhnet
{

bool BaseLayer::supportAvx      = false;
bool BaseLayer::supportFma      = false;
bool BaseLayer::isPreviewMode   = false;

void BaseLayer::initSimd()
{
#ifdef USE_X86
    SimdInfo info;
    info.checkSimd();

    supportAvx = info.getSupportAVX2();
    supportFma = info.getSupportFMA3();

    std::cout<<"checking simd."<<std::endl;

    if(supportAvx&&supportFma)
    {
        std::cout<<"avx2 speed up"<<std::endl<<std::endl;
    }
#endif
}

BaseLayer::BaseLayer()
{

}

BaseLayer::~BaseLayer()
{
    releaseArr(output);
}

void BaseLayer::setPreviewMode(const bool &previewMode)
{
    BaseLayer::isPreviewMode = previewMode;
}

void BaseLayer::forward(NetworkState &netState)
{
    (void)netState;
}

void BaseLayer::loadAllWeigths(std::vector<float> &weights)
{
    (void)weights;
}
}
