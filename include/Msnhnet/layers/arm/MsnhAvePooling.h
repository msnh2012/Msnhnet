#ifndef MSNHNETAvePoolingLAYERARM_H
#define MSNHNETAvePoolingLAYERARM_H
#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class MsnhNet_API AvePoolingLayerArm
{
public:
    static void pooling(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, 
                        float* &dest, const int& ceilModel);
};

}
#endif
#endif //MSNHNETBATCHNORMLAYERARM_H
