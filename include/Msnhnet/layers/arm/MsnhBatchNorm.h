#ifndef MSNHNETBATCHNORMLAYERARM_H
#define MSNHNETBATCHNORMLAYERARM_H
#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class BatchNormLayerArm
{
public:
    static void BatchNorm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float* dest,
                          float *const &Scales, float *const &rollMean, float *const &rollVariance, float *const &biases);
};

}
#endif
#endif //MSNHNETBATCHNORMLAYERARM_H
