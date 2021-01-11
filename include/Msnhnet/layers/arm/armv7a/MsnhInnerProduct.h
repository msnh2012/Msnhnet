#ifndef MSNHNETINNERPRODUCT_H
#define MSNHNETINNERPRODUCT_H
#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class MsnhNet_API InnerProductArm
{
public:
    void InnerProduct(float *const &src,  const int &inChannel,  float *const &weight, float* &dest, const int& outChannel);
};

}
#endif
#endif
