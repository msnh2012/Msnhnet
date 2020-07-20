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
    SoftMaxLayer(const int &batch, const int &inputNum, const int &groups);
    int         groups              =   0;
    float       temperature         =   0;

    virtual void forward(NetworkState &netState);
};
}

#endif 
