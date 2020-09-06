#ifndef MSNHNETVIEW_H
#define MSNHNETVIEW_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/utils/MsnhExport.h"
namespace Msnhnet
{

class MsnhNet_API ViewLayer : public BaseLayer
{
public:
    ViewLayer(const int &batch, const int &width, const int &height, const int &channel, const int &outWidth, const int &outHeight, const int &outChannel);
    ~ViewLayer();

    virtual  void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif
};

}

#endif 

