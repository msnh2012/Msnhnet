#ifndef MSNHLOCALAVGPOOL_H
#define MSNHLOCALAVGPOOL_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API LocalAvgPoolLayer : public BaseLayer
{
public:
    LocalAvgPoolLayer(const int &batch, const int &height, const int &width, const int &channel, const int &kSizeX, const int &kSizeY,
                      const int &strideX, const int &strideY, const int &paddingX, const int &paddingY, const int&ceilMode, const int &antialiasing);

   int         kSizeX              =   0;
    int         kSizeY              =   0;
    int         stride              =   0;
    int         strideX             =   0;
    int         strideY             =   0;
    int         paddingX            =   0;
    int         paddingY            =   0;

   int         ceilMode            =   0;

   int         antialiasing        =   0;

   virtual void forward(NetworkState &netState);

   ~LocalAvgPoolLayer();
};
}

#endif 

