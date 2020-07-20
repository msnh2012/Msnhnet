#ifndef MSNHPADDINGLAYER_H
#define MSNHPADDINGLAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API PaddingLayer : public BaseLayer
{
public:
    PaddingLayer(const int &batch,  const int &height, const int &width, const int &channel, const int &top,
                     const int &down, const int &left, const int &right, const float &paddingVal);

    ~PaddingLayer(){}

    int     top         =   0;
    int     down        =   0;
    int     left        =   0;
    int     right       =   0;
    float   paddingVal  =   0;

    virtual void forward(NetworkState &netState);

};
}

#endif 
