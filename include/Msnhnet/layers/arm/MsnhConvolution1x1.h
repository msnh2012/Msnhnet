#ifndef MSNHNETCONVOLUTIONALLAYERARM1X1_H
#define MSNHNETCONVOLUTIONALLAYERARM1X1_H
#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class MsnhNet_API ConvolutionalLayerArm1x1
{
public:
    //bottom: src, inWidth, inHeight, inChannel
    //top: dest, outWidth, outHeight, outChannel
    static void conv1x1s1Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel);
    static void conv1x1s2Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel);
    static void conv1x1s1SgemmTransformKenel(float *const &kernel, float* &dest, const int &inChannel, const int &outChannel);
    
    static void conv1x1s1SgemmNeon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel);
};

}
#endif
#endif //MSNHNETCONVOLUTIONALLAYERARM1X1_H
