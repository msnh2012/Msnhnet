#ifndef MSNHREDUCTIONLAYER_H
#define MSNHREDUCTIONLAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/io/MsnhParser.h"
#include "Msnhnet/core/MsnhBlas.h"

namespace Msnhnet
{

class ReductionLayer : public BaseLayer
{
public:
    ReductionLayer(const int &batch,  const int &height, const int &width, const int &channel, const int &axis, const ReductionType &reductionType);
    ~ReductionLayer();

    virtual void forward(NetworkState &netState);

    virtual void mallocMemory();

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    ReductionType getReductionType() const;

    int getAxis() const;

private:
    int _axis    =   -1;
    ReductionType  _reductionType;
};

}
#endif 

