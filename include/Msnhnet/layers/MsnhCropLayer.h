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
    CropLayer(const int &_batch, const int &_height, const int &_width, const int &_channel,
              const int &cropHeight, const int &cropWidth, const int &_flip, const float &angle,
              const float &_saturation, const float &_exposure);

    virtual void forward(NetworkState &netState);
    void resize(const int &_width, const int &_height);

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

