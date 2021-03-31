#ifdef USE_OPENCL

#include "Msnhnet/layers/opencl/MsnhBatchNormCL.h"


namespace Msnhnet
{
    void BatchNormCL::batchNormCL(
        cl_mem &src,
        cl_mem &dst, 
        cl_mem &biases, 
        cl_mem &Scales, 
        cl_mem &rollMean, 
        cl_mem &rollVariance, 
        const int &inWidth, 
        const int &inHeight,  
        const int &inChannel, 
        cl_kernel &kernel)
    {

        std::cout << "batchNorm opencl execution" << std::endl;


        cl_int status = 0;
        int inSize = inWidth * inHeight;

        status = clSetKernelArg(kernel, 0, sizeof(int), (void*) &inSize);
        status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &src);
        status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &biases);
        status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) &Scales);
        status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*) &rollMean);
        status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*) &rollVariance);
        status |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*) &dst);
        CHECKSTATUS(status, "set kernel args");

        size_t global[1] = {inChannel};
        size_t local[1] = {1};
        cl_event eventPoint;
        status |= clEnqueueNDRangeKernel(clScheduler::get().queue(), kernel, 1, NULL, global, local, 0, NULL, &eventPoint);
        clWaitForEvents(1, &eventPoint);
        clReleaseEvent(eventPoint);

    }
}



#endif
