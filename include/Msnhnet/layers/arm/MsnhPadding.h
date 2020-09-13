#ifndef MSNHPADDINGLAYERARM_H
#define MSNHPADDINGLAYERARM_H

#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{

class MsnhNet_API PaddingLayerArm
{
public:
    //bottom: src, inWidth, inHeight, inChannel
    //top: dest, outWidth, outHeight, outChannel
    void padding(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, 
                                        float* &dest, const int &top, const int &down, const int &left, const int &right, const int& val);
};

}
#endif
#endif
