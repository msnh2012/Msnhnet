#ifndef MSNHUPSAMPLE_H
#define MSNHUPSAMPLE_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API UpSampleLayer : public BaseLayer
{
public:
    UpSampleLayer(const int &batch, const int &width, const int &height, const int &channel, const int &stride, const float &scale);
    int         reverse     =   0;
    int         stride      =   0;
    float       scale       =   1.f;

   virtual void forward(NetworkState &netState);
    void resize(const int &width, const int &height);
};
}

#endif 

