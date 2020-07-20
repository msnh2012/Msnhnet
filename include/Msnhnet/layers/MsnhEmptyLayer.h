#ifndef MSNHEMPTYLAYER_H
#define MSNHEMPTYLAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API EmptyLayer : public BaseLayer
{
public:
    EmptyLayer(const int &batch, const int &width, const int &height, const int &channel);
    ~EmptyLayer();

    virtual  void forward(NetworkState &netState);
};
}

#endif 
