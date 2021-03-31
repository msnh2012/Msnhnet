#ifndef MSNHCONNECTEDCL_H
#define MSNHCONNECTEDCL_H

#ifdef USE_OPENCL
#include <CL/cl.h>
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{

class MsnhNet_API ConnectedCL
{
public:
    static void connectedCL(
        cl_mem &input, 
        const int &m, 
        const int &n,  
        const int &k, 
        cl_kernel &kernel,
        cl_mem &filter,
        cl_mem &dst);
                  
};

}
#endif
#endif
