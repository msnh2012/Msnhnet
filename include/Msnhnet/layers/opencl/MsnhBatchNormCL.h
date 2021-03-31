#ifndef MSNHBATCHNORMCL_H
#define MSNHBATCHNORMCL_H

#ifdef USE_OPENCL
#include <CL/cl.h>
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{

class MsnhNet_API BatchNormCL
{
public:
    static void batchNormCL(
        cl_mem &src,
        cl_mem &dst, 
        cl_mem &biases, 
        cl_mem &Scales, 
        cl_mem &rollMean, 
        cl_mem &rollVariance, 
        const int &inWidth, 
        const int &inHeight,  
        const int &inChannel, 
        cl_kernel &kernel);
                  
};

}
#endif
#endif
