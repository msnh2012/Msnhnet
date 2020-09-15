#ifndef MSNHNETBATCHNORMLAYERARM_H
#define MSNHNETBATCHNORMLAYERARM_H
#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class MsnhNet_API BatchNormLayerArm
{
public:
    static void BatchNorm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float* dest,
                          float *const &Scales, float *const &rollMean, float *const &rollVariance, float *const &biases, const float &eps);
};

}
#endif
#endif //MSNHNETBATCHNORMLAYERARM_H
