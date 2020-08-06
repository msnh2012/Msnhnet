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
    PaddingLayer(const int &_batch,  const int &_height, const int &_width, const int &_channel, const int &_top,
                     const int &_down, const int &_left, const int &_right, const float &_paddingVal);

    ~PaddingLayer(){}

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

