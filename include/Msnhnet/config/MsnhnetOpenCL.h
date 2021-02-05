#ifndef MSNHNET_OPENCL_H
#define MSNHNET_OPENCL_H
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <string>
#include <utility>
#include <iostream>

/** Class for loading OpenCL symbols. */
class CLSymbols final
{
public:
    /** Default Constructor */
    CLSymbols() noexcept(false);
    /** Load OpenCL symbols from handle
     *
     * @param[in] handle Handle to load symbols from
     */
    void load_symbols(void *handle);
    /** Get the static instance of CLSymbols.
     *
     * @return The static instance of CLSymbols.
     */
    static CLSymbols &get();
    /** Load symbols from the given OpenCL library path.
     *
     * @param[in] library Path to the OpenCL library.
     *
     * @return True if loading the library is successful.
     */
    bool load(const std::string &library);
    /** Load symbols from any of the default OpenCL library names.
     *
     * @return True if loading any library is successful.
     */
    bool load_default();

#define DECLARE_FUNCTION_PTR(func_name) \
    std::function<decltype(func_name)> func_name##_ptr = nullptr

    DECLARE_FUNCTION_PTR(clCreateContext);
    DECLARE_FUNCTION_PTR(clCreateContextFromType);
    DECLARE_FUNCTION_PTR(clCreateCommandQueue);
    DECLARE_FUNCTION_PTR(clGetContextInfo);
    DECLARE_FUNCTION_PTR(clBuildProgram);
    DECLARE_FUNCTION_PTR(clEnqueueNDRangeKernel);
    DECLARE_FUNCTION_PTR(clSetKernelArg);
    DECLARE_FUNCTION_PTR(clReleaseKernel);
    DECLARE_FUNCTION_PTR(clCreateProgramWithSource);
    DECLARE_FUNCTION_PTR(clCreateBuffer);
    DECLARE_FUNCTION_PTR(clRetainKernel);
    DECLARE_FUNCTION_PTR(clCreateKernel);
    DECLARE_FUNCTION_PTR(clGetProgramInfo);
    DECLARE_FUNCTION_PTR(clFlush);
    DECLARE_FUNCTION_PTR(clFinish);
    DECLARE_FUNCTION_PTR(clReleaseProgram);
    DECLARE_FUNCTION_PTR(clRetainContext);
    DECLARE_FUNCTION_PTR(clCreateProgramWithBinary);
    DECLARE_FUNCTION_PTR(clReleaseCommandQueue);
    DECLARE_FUNCTION_PTR(clEnqueueMapBuffer);
    DECLARE_FUNCTION_PTR(clRetainProgram);
    DECLARE_FUNCTION_PTR(clGetProgramBuildInfo);
    DECLARE_FUNCTION_PTR(clEnqueueReadBuffer);
    DECLARE_FUNCTION_PTR(clEnqueueWriteBuffer);
    DECLARE_FUNCTION_PTR(clReleaseEvent);
    DECLARE_FUNCTION_PTR(clReleaseContext);
    DECLARE_FUNCTION_PTR(clRetainCommandQueue);
    DECLARE_FUNCTION_PTR(clEnqueueUnmapMemObject);
    DECLARE_FUNCTION_PTR(clRetainMemObject);
    DECLARE_FUNCTION_PTR(clReleaseMemObject);
    DECLARE_FUNCTION_PTR(clGetDeviceInfo);
    DECLARE_FUNCTION_PTR(clGetDeviceIDs);
    DECLARE_FUNCTION_PTR(clGetMemObjectInfo);
    DECLARE_FUNCTION_PTR(clRetainEvent);
    DECLARE_FUNCTION_PTR(clGetPlatformIDs);
    DECLARE_FUNCTION_PTR(clGetKernelWorkGroupInfo);
    DECLARE_FUNCTION_PTR(clGetCommandQueueInfo);
    DECLARE_FUNCTION_PTR(clGetKernelInfo);
    DECLARE_FUNCTION_PTR(clGetEventProfilingInfo);
    // DECLARE_FUNCTION_PTR(clSVMAlloc);
    // DECLARE_FUNCTION_PTR(clSVMFree);
    // DECLARE_FUNCTION_PTR(clEnqueueSVMMap);
    // DECLARE_FUNCTION_PTR(clEnqueueSVMUnmap);
    // DECLARE_FUNCTION_PTR(clEnqueueMarker);
    DECLARE_FUNCTION_PTR(clWaitForEvents);

    // Third-party extensions
    DECLARE_FUNCTION_PTR(clImportMemoryARM);

#undef DECLARE_FUNCTION_PTR

private:
    std::pair<bool, bool> _loaded;
};

bool opencl_is_available();
#endif