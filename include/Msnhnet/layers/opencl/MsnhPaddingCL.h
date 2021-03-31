#ifndef MSNHPADDINGCL_H
#define MSNHPADDINGCL_H

#ifdef USE_OPENCL
#include <CL/cl.h>
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{

class MsnhNet_API PaddingCL
{
public:
    static void paddingCL(
        cl_mem &src, 
        const int &inWidth, 
        const int &inHeight,  
        const int &inChannel, 
        cl_kernel &kernel,
        cl_mem &dst, 
        const int &top, 
        const int &down, 
        const int &left, 
        const int &right, 
        const float &val);
                  
};

}
#endif
#endif
