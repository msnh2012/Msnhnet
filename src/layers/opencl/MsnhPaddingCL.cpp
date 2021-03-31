#ifdef USE_OPENCL

#include "Msnhnet/layers/opencl/MsnhPaddingCL.h"

#include <fstream> ///////////////////////////////////////////////////


namespace Msnhnet
{
    void PaddingCL::paddingCL(
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
        const float &val)
    {

        std::cout << "maxPoolingGeneral" << std::endl;
        cl_int status = 0;

        status = clSetKernelArg(kernel, 0, sizeof(int), (void*) &inWidth);
        status |= clSetKernelArg(kernel, 1, sizeof(int), (void*) &inHeight);
        status |= clSetKernelArg(kernel, 2, sizeof(int), (void*) &top);
        status |= clSetKernelArg(kernel, 3, sizeof(int), (void*) &down);
        status |= clSetKernelArg(kernel, 4, sizeof(int), (void*) &left);
        status |= clSetKernelArg(kernel, 5, sizeof(int), (void*) &right);
        status |= clSetKernelArg(kernel, 6, sizeof(float), (void*) &val);
        status |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*) &src);
        status |= clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*) &dst);
        CHECKSTATUS(status, "set kernel args");

        size_t global[1] = {inChannel};
        size_t local[1] = {1};
        cl_event eventPoint;
        status |= clEnqueueNDRangeKernel(clScheduler::get().queue(), kernel, 1, NULL, global, local, 0, NULL, &eventPoint);
        clWaitForEvents(1, &eventPoint);
        clReleaseEvent(eventPoint);

        CHECKSTATUS(status, "max pool tensor");
    }
}



#endif
