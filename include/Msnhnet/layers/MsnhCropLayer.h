#ifndef MSNHCROPLAYER_H
#define MSNHCROPLAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhGemm.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API CropLayer : public BaseLayer
{
public:
    CropLayer(const int &batch, const int &height, const int &width, const int &channel,
              const int &cropHeight, const int &cropWidth, const int &flip, const float &angle,
              const float &saturation, const float &exposure);

   int         kSize               =   0;
    float       scale               =   0;
    int         flip                =   0;
    float       saturation          =   0;
    float       exposure            =   0;
    int         noAdjust            =   0;

   virtual void forward(NetworkState &netState);
    void resize(const int &width, const int &height);
};
}

#endif 

