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

    virtual void mallocMemory();
    virtual void forward(NetworkState &netState);
    void resize(const int &width, const int &height);

    int getKSize() const;

    float getScale() const;

    int getFlip() const;

    float getSaturation() const;

    float getExposure() const;

    int getNoAdjust() const;

protected:
    int         _kSize               =   0;
    float       _scale               =   0;
    int         _flip                =   0;
    float       _saturation          =   0;
    float       _exposure            =   0;
    int         _noAdjust            =   0;
};
}

#endif 

