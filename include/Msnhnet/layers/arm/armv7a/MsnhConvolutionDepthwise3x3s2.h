#ifndef MSNHNETCONVOLUTIONALDEPTHWISELAYERARM3X3S2_H
#define MSNHNETCONVOLUTIONALDEPTHWISELAYERARM3X3S2_H
#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class MsnhNet_API ConvolutionalDepthwiseLayerArm3x3s2
{
public:
    //bottom: src, inWidth, inHeight, inChannel
    //top: dest, outWidth, outHeight, outChannel
    static void convdepthwise3x3s2Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel);
};

}
#endif
#endif //MSNHNETCONVOLUTIONALDEPTHWISELAYERARM3X3S1_H
