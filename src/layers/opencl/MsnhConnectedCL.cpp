#ifdef USE_OPENCL

#include "Msnhnet/layers/opencl/MsnhConnectedCL.h"

#include <fstream> ///////////////////////////////////////////////////


namespace Msnhnet
{
    void ConnectedCL::connectedCL(
        cl_mem &input, 
        const int &m, 
        const int &n,  
        const int &k, 
        cl_kernel &kernel,
        cl_mem &filter,
        cl_mem &dst)
    {

        std::cout << "connected opencl" << std::endl;
        cl_int status = 0;

        status = clSetKernelArg(kernel, 0, sizeof(int), (void*) &m);
        status |= clSetKernelArg(kernel, 1, sizeof(int), (void*) &n);
        status |= clSetKernelArg(kernel, 2, sizeof(int), (void*) &k);
        status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &input);
        status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*) &filter);
        status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*) &dst);
        CHECKSTATUS(status, "set kernel args");

        size_t global[2] = {m, n};
        cl_event eventPoint;
        status |= clEnqueueNDRangeKernel(clScheduler::get().queue(), kernel, 2, NULL, global, NULL, 0, NULL, &eventPoint);
        clWaitForEvents(1, &eventPoint);
        clReleaseEvent(eventPoint);

        CHECKSTATUS(status, "calc matrix mul");
    }
}



#endif
