#ifndef MSNHPADDINGLAYER_H
#define MSNHPADDINGLAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhPaddingLayerGPU.h"
#endif
namespace Msnhnet
{
class MsnhNet_API PaddingLayer : public BaseLayer
{
public:
    PaddingLayer(const int &batch,  const int &height, const int &width, const int &channel, const int &top,
                     const int &down, const int &left, const int &right, const float &paddingVal);

    ~PaddingLayer(){}

    virtual void mallocMemory();

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    int getTop() const;

    int getDown() const;

    int getLeft() const;

    int getRight() const;

    float getPaddingVal() const;

protected:
    int     _top         =   0;
    int     _down        =   0;
    int     _left        =   0;
    int     _right       =   0;
    float   _paddingVal  =   0;

};
}

#endif 

