#ifndef MSNHNETABSVALLAYERARM_H
#define MSNHNETABSVALLAYERARM_H
#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class MsnhNet_API AbsValLayerArm
{
public:
    static void AbsVal(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float* dest);
    static void AbsValInplace(float* src, const int &inWidth, const int &inHeight,  const int &inChannel);
};

}
#endif
#endif //MSNHNETBATCHNORMLAYERARM_H
