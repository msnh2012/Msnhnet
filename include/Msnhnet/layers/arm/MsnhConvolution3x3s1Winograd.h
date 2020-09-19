#ifndef MSNHNETCONVOLUTIONALLAYERARM3X3S1WINOGRAD_H
#define MSNHNETCONVOLUTIONALLAYERARM3X3S1WINOGRAD_H
#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class MsnhNet_API ConvolutionalLayerArm3x3s1Winograd
{
public:
    //bottom: src, inWidth, inHeight, inChannel
    //top: dest, outWidth, outHeight, outChannel
    static void conv3x3s1WinogradTransformKenel(float *const &kernel, float* &kernel_tm, const int &inChannel, const int &outChannel);
    static void conv3x3s1WinogradNeon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel);

};

}
#endif
#endif //MSNHNETCONVOLUTIONALLAYERARM3X3S1_H
