#ifdef USE_OPENCL

#include "Msnhnet/layers/opencl/MsnhActivationsCL.h"
#include "Msnhnet/config/MsnhnetCfg.h"


namespace Msnhnet
{
    void ActivationsCL::activateArrayCL(float *x, const int &numX, cl_kernel &kernel, const float &param){

        cl_int status = 0;
        cl_mem dst = clCreateBuffer(clScheduler::get().context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numX * sizeof(float), x, &status);


        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &dst);

        // size_t global[1] = {(numX + 3) / 4 };
        size_t global[1] = {numX};
        cl_event eventPoint;
        status |= clEnqueueNDRangeKernel(clScheduler::get().queue(), kernel, 1, NULL, global, NULL, 0, NULL, &eventPoint);
        clWaitForEvents(1, &eventPoint);
        clReleaseEvent(eventPoint);

        status |= clEnqueueReadBuffer(clScheduler::get().queue(), dst, CL_TRUE, 0, numX * sizeof(float), x, 0, NULL, NULL);
    }
}

#endif