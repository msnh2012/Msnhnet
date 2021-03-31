#ifndef MSNHCONVOLUTIONSGEMMCL_H
#define MSNHCONVOLUTIONSGEMMCL_H

#ifdef USE_OPENCL
#include <CL/cl.h>
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/opencl/MsnhPaddingCL.h"

namespace Msnhnet
{

class MsnhNet_API ConvolutionSgemmCL
{
public:
    static void convolutionIm2colSgemmCL(
        cl_mem &src, 
        const int &inWidth, 
        const int &inHeight,  
        const int &inChannel, 
        cl_mem &filter, 
        cl_kernel &kernel_cov, 
        cl_kernel &kernel_im2col,
        const int &filterW, 
        const int &filterH, 
        cl_mem &dst, 
        const int &outWidth, 
        const int &outHeight, 
        const int &outChannel, 
        const int& StrideH, 
        const int &StrideW, 
        const int &paddingX, 
        const int &paddingY, 
        const int &dilationW, 
        const int &dilationH);

    static void conv3x3s1WinogradTransformKenelCL(cl_kernel &kernel, cl_mem &filter, cl_mem &filterWino,const int &inChannel, const int &outChannel);
    static void conv3x3s1WinogradCL(
        cl_mem &src, 
        const int &inWidth, 
        const int &inHeight,  
        const int &inChannel, 
        cl_mem &filter, 
        cl_kernel &kernel_cov, 
        cl_kernel &kernel_pad,
        cl_mem &dst, 
        const int &outWidth, 
        const int &outHeight, 
        const int &outChannel, 
        const int &paddingX, 
        const int &paddingY);
    
    inline static int is_a_ge_zero_and_a_lt_b_cl(const int &a, const int &b)
    {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }
                            
    // static void convolutionTransformKernel(float *const &kernel, const int &kernelW, const int &kernelH, float* &dest, const int &inChannel,
    //                         const int &outChannel);                        
};

}
#endif
#endif
