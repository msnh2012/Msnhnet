#ifndef MSNHSLICELAYER_H
#define MSNHSLICELAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhSliceLayerGPU.h"
#endif
namespace Msnhnet
{
class MsnhNet_API SliceLayer: public BaseLayer
{
public:
    SliceLayer(const int &batch, const int &height, const int &width, const int &channel, const int &getStart0, const int &step0,
               const int &start1, const int &step1, const int &start2, const int &step2);

    virtual void mallocMemory();
    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    int getStart0() const;

    int getStep0() const;

    int getStart1() const;

    int getStep1() const;

    int getStart2() const;

    int getStep2() const;

private:
    int _start0     =   0;
    int _step0      =   1;
    int _start1     =   0;
    int _step1      =   1;
    int _start2     =   0;
    int _step2      =   1;
};
}

#endif 

