#ifndef MSNHROUTELAYER_H
#define MSNHROUTELAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API RouteLayer : public BaseLayer
{
public:
    RouteLayer(const int &batch, std::vector<int> &inputLayerIndexes, std::vector<int> &inputLayerOutputs,
               const int &groups, const int &groupId);

   std::vector<int>inputLayerIndexes;
    std::vector<int>inputLayerOutputs;
    int         groups              =   0;
    int         groupIndex          =   0;

   virtual void forward(NetworkState &netState);
    void resize(Network &net);
};
}

#endif 

