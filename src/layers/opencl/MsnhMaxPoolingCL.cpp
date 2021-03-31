#ifdef USE_OPENCL

#include "Msnhnet/layers/opencl/MsnhMaxPoolingCL.h"

#include <fstream> ///////////////////////////////////////////////////


namespace Msnhnet
{
    void MaxPoolingCL::maxPoolingGeneral(
        float *const &src, 
        const int &inWidth, 
        const int &inHeight,  
        const int &inChannel, 
        cl_kernel &kernel,
        const int &filterW, 
        const int &filterH, 
        float* dst, 
        const int &outWidth, 
        const int &outHeight, 
        const int &outChannel, 
        const int &strideX, 
        const int &strideY, 
        const int &paddingX, 
        const int &paddingY)
    {

        std::cout << "maxPoolingGeneral" << std::endl;


        cl_int status = 0;
        cl_mem mem_src = clCreateBuffer(clScheduler::get().context(),  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inWidth * inHeight * inChannel * sizeof(float), src, &status);
        CHECKSTATUS(status, "create mem_src");
        cl_mem mem_dst = clCreateBuffer(clScheduler::get().context(),  CL_MEM_WRITE_ONLY, outWidth * outHeight * outChannel * sizeof(float), NULL, &status);
        CHECKSTATUS(status, "create mem_dst");
        

        status = clSetKernelArg(kernel, 0, sizeof(int), (void*) &inWidth);
        status |= clSetKernelArg(kernel, 1, sizeof(int), (void*) &inHeight);
        status |= clSetKernelArg(kernel, 2, sizeof(int), (void*) &inChannel);
        status |= clSetKernelArg(kernel, 3, sizeof(int), (void*) &outWidth);
        status |= clSetKernelArg(kernel, 4, sizeof(int), (void*) &outHeight);
        status |= clSetKernelArg(kernel, 5, sizeof(int), (void*) &outChannel);
        status |= clSetKernelArg(kernel, 6, sizeof(int), (void*) &filterW);
        status |= clSetKernelArg(kernel, 7, sizeof(int), (void*) &filterH);
        status |= clSetKernelArg(kernel, 8, sizeof(int), (void*) &strideX);
        status |= clSetKernelArg(kernel, 9, sizeof(int), (void*) &strideY);
        status |= clSetKernelArg(kernel, 10, sizeof(int), (void*) &paddingX);
        status |= clSetKernelArg(kernel, 11, sizeof(int), (void*) &paddingY);
        status |= clSetKernelArg(kernel, 12, sizeof(cl_mem), (void*) &mem_src);
        status |= clSetKernelArg(kernel, 13, sizeof(cl_mem), (void*) &mem_dst);
        CHECKSTATUS(status, "set kernel args");

        size_t global[3] = {outWidth, outHeight, outChannel};
        size_t local[3] = {1, 1, 1};
        cl_event eventPoint;
        status |= clEnqueueNDRangeKernel(clScheduler::get().queue(), kernel, 3, NULL, global, local, 0, NULL, &eventPoint);
        clWaitForEvents(1, &eventPoint);
        clReleaseEvent(eventPoint);

        CHECKSTATUS(status, "max pool tensor");
        status |= clEnqueueReadBuffer(clScheduler::get().queue(), mem_dst, CL_TRUE, 0, outWidth * outHeight * outChannel * sizeof(float), dst, 0, NULL, NULL);

    }
}



#endif
