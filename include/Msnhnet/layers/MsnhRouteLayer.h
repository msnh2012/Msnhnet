#ifndef MSNHROUTELAYER_H
#define MSNHROUTELAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/layers/MsnhActivations.h"
namespace Msnhnet
{
class MsnhNet_API RouteLayer : public BaseLayer
{
public:
    RouteLayer(const int &batch, std::vector<int> &inputLayerIndexes, std::vector<int> &inputLayerOutputs,
               const int &groups, const int &groupId, const int &addModel, const ActivationType &activation, std::vector<float> actParams);

    virtual void forward(NetworkState &netState);

    virtual void mallocMemory();

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    void resize(Network &net);

    std::vector<int> getInputLayerIndexes() const;

    std::vector<int> getInputLayerOutputs() const;

    int getGroups() const;

    int getGroupIndex() const;

    int getAddModel() const;

protected:
    std::vector<int> _inputLayerIndexes;
    std::vector<int> _inputLayerOutputs;
    int         _groups              =   0;
    int         _groupIndex          =   0;
    int         _addModel            =   0;

};
}

#endif 

