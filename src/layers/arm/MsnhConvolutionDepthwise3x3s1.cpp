#define USE_ARM 1
#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolutionDepthwise3x3s1.h"
namespace Msnhnet
{
    void ConvolutionalDepthwiseLayerArm3x3s1::convdepthwise3x3s1Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        int ccOutChannel = outChannel >> 1;
        int ccRemainOutChannel = ccOutChannel << 1;

        const int in_size = inWidth * inHeight;
        const int out_size = outWidth * outHeight;
        const int group = inChannel;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif 
        for(int g = 0; g < group; g++){
            float *dest0 = dest + g * out_size;
            const float* kernel0 = kernel + g * 9;
            

        }
    }
}

#endif