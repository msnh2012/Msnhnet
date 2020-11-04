#define USE_ARM 1
#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolution1x1.h"

namespace Msnhnet
{
    void ConvolutionalLayerArm1x1::conv1x1Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        
        
    }    
}

#endif