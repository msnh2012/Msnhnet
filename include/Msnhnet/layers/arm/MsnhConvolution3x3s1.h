#ifndef CONVOLUTIONALLAYERARM3X3S1
#define CONVOLUTIONALLAYERARM3X3S1
#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class MsnhNet_API ConvolutionalLayerArm3x3s1
{
public:
    //bottom: src, inWidth, inHeight, inChannel
    //top: dest, outWidth, outHeight, outChannel
    static void conv3x3s1Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel);
};

}
#endif
#endif //CONVOLUTIONALLAYERARM3X3S1
