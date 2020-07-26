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
    SoftMaxLayer(const int &_batch, const int &_inputNum, const int &_groups, const float &_temperature);
    virtual void forward(NetworkState &netState);

    int getGroups() const;

    float getTemperature() const;

protected:
    int         _groups              =   0;
    float       _temperature         =   0;
};
}

#endif 

