#ifndef MSNHSOFTMAX_H
#define MSNHSOFTMAX_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API SoftMaxLayer : public BaseLayer
{
public:
    SoftMaxLayer(const int &batch, const int &inputNum, const int &groups, const float &temperature);

    ~SoftMaxLayer();

    virtual void forward(NetworkState &netState);

    virtual void mallocMemory();

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    int getGroups() const;

    float getTemperature() const;

protected:
    int         _groups              =   0;
    float       _temperature         =   0;

#ifdef USE_GPU
#ifdef USE_CUDNN

    cudnnTensorDescriptor_t         _inputDesc;
    cudnnTensorDescriptor_t         _outputDesc;
#endif
#endif
};
}

#endif 

