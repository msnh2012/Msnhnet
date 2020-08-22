#ifndef MSNHPREMUTELAYER_H
#define MSNHPREMUTELAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/core/MsnhBlas.h"
#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhPermuteLayerGPU.h"
#endif

namespace Msnhnet
{

class MsnhNet_API PermuteLayer : public BaseLayer
{
public:
    PermuteLayer(const int &batch,  const int &height, const int &width, const int &channel, const int &getDim0, const int &dim1, const int &dim2);
    ~PermuteLayer();

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    int getDim0() const;
    int getDim1() const;
    int getDim2() const;

private:
    int _dim0   = 0;
    int _dim1   = 0;
    int _dim2   = 0;
};

}

#endif 

