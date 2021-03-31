#ifndef ACTIVATIONLAYER_H
#define ACTIVATIONLAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/layers/MsnhActivations.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API ActivationLayer:public BaseLayer
{
public:
    ActivationLayer(const int &batch, const int &width, const int &height, const int &channel, const int &inputNum, const ActivationType &activation, const std::vector<float> &actParams);
    ~ActivationLayer();

    virtual void forward(NetworkState &netState);
    virtual void mallocMemory();
    virtual void loadAllWeigths(std::vector<float> &weights);
    virtual void saveAllWeights(const int &mainIdx, const int &branchIdx=-1, const int &branchIdx1=-1);

    void loadPreluWeights(float *const &weights, const int& len);
#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif


protected:
    float *_preluWeights    = nullptr;
#ifdef USE_GPU
    float *_gpuPreluWeights = nullptr;
#endif
    int _nPreluWeights      = 0;
};
}

#endif 

