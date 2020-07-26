#ifndef MSNHGLOBALAVGPOOL_H
#define MSNHGLOBALAVGPOOL_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{

class MsnhNet_API GlobalAvgPoolLayer: public BaseLayer
{
public:
    GlobalAvgPoolLayer(const int &_batch, const int &_height, const int &_width, const int &_channel);
    ~GlobalAvgPoolLayer();

    virtual void forward(NetworkState &netState);
};

}

#endif 

