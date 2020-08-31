#include "Msnhnet/layers/arm/MsnhConvolutionSgemm.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    void ConvolutionLayerSgemm::convolution_im2col_sgemm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                            const int &kernelW, const int &kernelH,int float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        // 1. im2col

        float *src_im2col = new float[outWidth * outHeight * kernelH * kernelW * inChannel];
        
        const int stride = kernelW * kernelH * outHeight * outWidth;

        

    }
}