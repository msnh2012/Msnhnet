#ifndef MSNHNETBIASLAYERARM_H
#define MSNHNETBIASLAYERARM_H
#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class MsnhNet_API BiasLayerArm
{
public:
    static void Bias(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &bias,float* dest);
    static void BiasInplace(float* src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &bias);
};

}
#endif
#endif //MSNHNETBATCHNORMLAYERARM_H
