#ifndef MSNHNETCONVOLUTIONLAYERSGEMM_H
#define MSNHNETCONVOLUTIONLAYERSGEMM_H

#ifdef USE_ARM
#include "Msnhnet/config/MsnhnetCfg.h"
namespace Msnhnet
{

class MsnhNet_API ConvolutionLayerSgemm
{
public:
    //bottom: src, inWidth, inHeight, inChannel
    //top: dest, outWidth, outHeight, outChannel
    static void convolutionIm2colSgemm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel, float *const kernel_im2col_pack,
                            const int &kernelW, const int &kernelH, float* &dest, const int &outWidth, const int &outHeight, const int &outChannel,
                            const int& StrideH, const int &StrideW);
    
    static void convolutionTransformKernel(float *const &kernel, const int &kernelW, const int &kernelH, float* &dest, const int &inChannel,
                            const int &outChannel);
};

}
#endif
#endif
