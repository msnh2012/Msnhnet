#ifdef USE_OPENCL

#include "Msnhnet/layers/opencl/MsnhLocalAvgPoolCL.h"

#include <fstream> ///////////////////////////////////////////////////


namespace Msnhnet
{
    void LocalAvgPoolCL::localAvgPool(
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
        const int &paddingY)
    {

        std::cout << "LocalAveragePooling" << std::endl;
        cl_int status = 0;



        status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &inWidth);
        status |= clSetKernelArg(kernel, 1, sizeof(cl_int), (void*) &inHeight);
        status |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void*) &outWidth);
        status |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*) &outHeight);
        status |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void*) &filterW);
        status |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void*) &filterH);
        status |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void*) &strideX);
        status |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void*) &strideY);
        status |= clSetKernelArg(kernel, 8, sizeof(cl_int), (void*)&paddingX);
        status |= clSetKernelArg(kernel, 9, sizeof(cl_int), (void*)&paddingY);
        status |= clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*) &src);
        status |= clSetKernelArg(kernel, 11, sizeof(cl_mem), (void*) &dst);
        CHECKSTATUS(status, "set kernel args");

        size_t global[1] = {outChannel};
        size_t local[1] = {1};
        cl_event eventPoint;
        status |= clEnqueueNDRangeKernel(clScheduler::get().queue(), kernel, 1, NULL, global, local, 0, NULL, &eventPoint);
        clWaitForEvents(1, &eventPoint);
        clReleaseEvent(eventPoint);

        CHECKSTATUS(status, "local average pool tensor");

    }
}



#endif
