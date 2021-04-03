#ifndef MSNHLOCALPOOLCL_H
#define MSNHLOCALPOOLCL_H

#ifdef USE_OPENCL
#include <CL/cl.h>
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{

class MsnhNet_API LocalAvgPoolCL
{
public:
    static void localAvgPool(
        cl_mem &src, 
        const int &inWidth, 
        const int &inHeight,  
        const int &inChannel, 
        cl_kernel &kernel,
        const int &filterW, 
        const int &filterH, 
        cl_mem &dst, 
        const int &outWidth, 
        const int &outHeight, 
        const int &outChannel, 
        const int &strideX, 
        const int &strideY, 
        const int &paddingX, 
        const int &paddingY);
                  
};

}
#endif
#endif
