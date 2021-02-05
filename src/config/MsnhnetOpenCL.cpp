#include "compute_opencl.h"
#include <vector>
#include <dlfcn.h>

CLSymbols::CLSymbols() noexcept(false)
    : _loaded(
        {false, false})
{
}

CLSymbols &CLSymbols::get()
{
    static CLSymbols symbols;
    return symbols;
}

bool CLSymbols::load_default()
{
    static const std::vector<std::string> libraries{ "libOpenCL.so", "libGLES_mali.so", "libmali.so" };

// #ifdef __aarch64__
// #define MALI_LIB_PATH "/vendor/lib64/egl/libGLES_mali.so"
// #else
// #define MALI_LIB_PATH "/vendor/lib/egl/libGLES_mali.so"
// #endif

    // static const std::vector<std::string>
    if(_loaded.first)
    {
        return _loaded.second;
    }

    // Indicate that default loading has been tried
    _loaded.first = true;

    // if (load(MALI_LIB_PATH)){
    //     if (this->clBuildProgram_ptr == NULL) {
    //             // LOGE("Failed to load OpenCL symbols from shared library\n");
    //             std::cout << "Failed to load OpenCL symbols from shared library\n" << std::endl;
    //     }
    //     return true;
    // }

    for(const auto &lib : libraries)
    {
        if(load(lib))
        {
            if (this->clBuildProgram_ptr == NULL) {
                // LOGE("Failed to load OpenCL symbols from shared library\n");
                
                std::cout << "Failed to load OpenCL symbols from shared library\n" << std::endl;
            }
            return true;
        }
    }

    // LOGE("Couldn't find any OpenCL library.\n");
    std::cout << "Couldn't find any OpenCL library.\n" << std::endl;
    return false;
}

bool CLSymbols::load(const std::string &library)
{
    std::cout << "load lib.so " << library << std::endl;
    void *handle = dlopen(library.c_str(), RTLD_LAZY | RTLD_LOCAL);

    if(handle == nullptr)
    {
        // LOGE("Can't load %s : %s \n", library.c_str(), dlerror());
        std::cout << "Can't load " << library.c_str() << " : " << dlerror() << " \n" << std::endl;
        // Set status of loading to failed
        _loaded.second = false;
        return false;
    }

#define LOAD_FUNCTION_PTR(func_name, handle) \
    func_name##_ptr = reinterpret_cast<decltype(func_name) *>(dlsym(handle, #func_name));

    LOAD_FUNCTION_PTR(clCreateContext, handle);
    LOAD_FUNCTION_PTR(clCreateContextFromType, handle);
    LOAD_FUNCTION_PTR(clCreateCommandQueue, handle);
    LOAD_FUNCTION_PTR(clGetContextInfo, handle);
    LOAD_FUNCTION_PTR(clBuildProgram, handle);
    LOAD_FUNCTION_PTR(clEnqueueNDRangeKernel, handle);
    LOAD_FUNCTION_PTR(clSetKernelArg, handle);
    LOAD_FUNCTION_PTR(clReleaseKernel, handle);
    LOAD_FUNCTION_PTR(clCreateProgramWithSource, handle);
    LOAD_FUNCTION_PTR(clCreateBuffer, handle);
    LOAD_FUNCTION_PTR(clRetainKernel, handle);
    LOAD_FUNCTION_PTR(clCreateKernel, handle);
    LOAD_FUNCTION_PTR(clGetProgramInfo, handle);
    LOAD_FUNCTION_PTR(clFlush, handle);
    LOAD_FUNCTION_PTR(clFinish, handle);
    LOAD_FUNCTION_PTR(clReleaseProgram, handle);
    LOAD_FUNCTION_PTR(clRetainContext, handle);
    LOAD_FUNCTION_PTR(clCreateProgramWithBinary, handle);
    LOAD_FUNCTION_PTR(clReleaseCommandQueue, handle);
    LOAD_FUNCTION_PTR(clEnqueueMapBuffer, handle);
    LOAD_FUNCTION_PTR(clRetainProgram, handle);
    LOAD_FUNCTION_PTR(clGetProgramBuildInfo, handle);
    LOAD_FUNCTION_PTR(clEnqueueReadBuffer, handle);
    LOAD_FUNCTION_PTR(clEnqueueWriteBuffer, handle);
    LOAD_FUNCTION_PTR(clReleaseEvent, handle);
    LOAD_FUNCTION_PTR(clReleaseContext, handle);
    LOAD_FUNCTION_PTR(clRetainCommandQueue, handle);
    LOAD_FUNCTION_PTR(clEnqueueUnmapMemObject, handle);
    LOAD_FUNCTION_PTR(clRetainMemObject, handle);
    LOAD_FUNCTION_PTR(clReleaseMemObject, handle);
    LOAD_FUNCTION_PTR(clGetDeviceInfo, handle);
    LOAD_FUNCTION_PTR(clGetDeviceIDs, handle);
    LOAD_FUNCTION_PTR(clGetMemObjectInfo, handle);
    LOAD_FUNCTION_PTR(clRetainEvent, handle);
    LOAD_FUNCTION_PTR(clGetPlatformIDs, handle);
    LOAD_FUNCTION_PTR(clGetKernelWorkGroupInfo, handle);
    LOAD_FUNCTION_PTR(clGetCommandQueueInfo, handle);
    LOAD_FUNCTION_PTR(clGetKernelInfo, handle);
    LOAD_FUNCTION_PTR(clGetEventProfilingInfo, handle);
    // LOAD_FUNCTION_PTR(clSVMAlloc, handle);
    // LOAD_FUNCTION_PTR(clSVMFree, handle);
    // LOAD_FUNCTION_PTR(clEnqueueSVMMap, handle);
    // LOAD_FUNCTION_PTR(clEnqueueSVMUnmap, handle);
    // LOAD_FUNCTION_PTR(clEnqueueMarker, handle);
    LOAD_FUNCTION_PTR(clWaitForEvents, handle);

    // Third-party extensions
    LOAD_FUNCTION_PTR(clImportMemoryARM, handle);

#undef LOAD_FUNCTION_PTR

    //Don't call dlclose(handle) or all the symbols will be unloaded !

    // Disable default loading and set status to successful
    _loaded = std::make_pair(true, true);

    return true;
}

bool opencl_is_available()
{
    CLSymbols::get().load_default();
    return CLSymbols::get().clBuildProgram_ptr != nullptr;
}


// cl_int clEnqueueMarker(cl_command_queue command_queue,
//                        cl_event        *event)
// {
//     CLSymbols::get().load_default();
//     auto func = CLSymbols::get().clEnqueueMarker_ptr;
//     if(func != nullptr)
//     {
//         return func(command_queue, event);
//     }
//     else
//     {
//         return CL_OUT_OF_RESOURCES;
//     }
// }

cl_int clWaitForEvents(cl_uint         num_events,
                       const cl_event *event_list)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clWaitForEvents_ptr;
    if(func != nullptr)
    {
        return func(num_events, event_list);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

// cl_int clEnqueueSVMMap(cl_command_queue command_queue, cl_bool blocking_map, cl_map_flags flags, void *svm_ptr,
//                        size_t size, cl_uint num_events_in_wait_list, const cl_event *event_wait_list, cl_event *event)
// {
//     CLSymbols::get().load_default();
//     auto func = CLSymbols::get().clEnqueueSVMMap_ptr;
//     if(func != nullptr)
//     {
//         return func(command_queue, blocking_map, flags, svm_ptr, size, num_events_in_wait_list, event_wait_list, event);
//     }
//     else
//     {
//         return CL_OUT_OF_RESOURCES;
//     }
// }

// cl_int clEnqueueSVMUnmap(cl_command_queue command_queue, void *svm_ptr, cl_uint num_events_in_wait_list,
//                          const cl_event *event_wait_list, cl_event *event)
// {
//     CLSymbols::get().load_default();
//     auto func = CLSymbols::get().clEnqueueSVMUnmap_ptr;
//     if(func != nullptr)
//     {
//         return func(command_queue, svm_ptr, num_events_in_wait_list, event_wait_list, event);
//     }
//     else
//     {
//         return CL_OUT_OF_RESOURCES;
//     }
// }

// void *clSVMAlloc(cl_context context, cl_svm_mem_flags_arm flags, size_t size, cl_uint alignment)
// {
//     CLSymbols::get().load_default();
//     auto func = CLSymbols::get().clSVMAlloc_ptr;
//     if(func != nullptr)
//     {
//         return func(context, flags, size, alignment);
//     }
//     else
//     {
//         return nullptr;
//     }
// }

// void clSVMFree(cl_context context, void *svm_pointer)
// {
//     CLSymbols::get().load_default();
//     auto func = CLSymbols::get().clSVMFree_ptr;
//     if(func != nullptr)
//     {
//         func(context, svm_pointer);
//     }
// }

cl_int clGetContextInfo(cl_context      context,
                        cl_context_info param_name,
                        size_t          param_value_size,
                        void           *param_value,
                        size_t         *param_value_size_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetContextInfo_ptr;
    if(func != nullptr)
    {
        return func(context, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_command_queue clCreateCommandQueue(cl_context                  context,
                                      cl_device_id                device,
                                      cl_command_queue_properties properties,
                                      cl_int                     *errcode_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clCreateCommandQueue_ptr;
    if(func != nullptr)
    {
        return func(context, device, properties, errcode_ret);
    }
    else
    {
        return nullptr;
    }
}

cl_context clCreateContext(
    const cl_context_properties *properties,
    cl_uint                      num_devices,
    const cl_device_id          *devices,
    void (*pfn_notify)(const char *, const void *, size_t, void *),
    void   *user_data,
    cl_int *errcode_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clCreateContext_ptr;
    if(func != nullptr)
    {
        return func(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
    }
    else
    {
        return nullptr;
    }
}

cl_context clCreateContextFromType(const cl_context_properties *properties,
                                   cl_device_type               device_type,
                                   void (*pfn_notify)(const char *, const void *, size_t, void *),
                                   void   *user_data,
                                   cl_int *errcode_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clCreateContextFromType_ptr;
    if(func != nullptr)
    {
        return func(properties, device_type, pfn_notify, user_data, errcode_ret);
    }
    else
    {
        return nullptr;
    }
}

cl_int clBuildProgram(
    cl_program          program,
    cl_uint             num_devices,
    const cl_device_id *device_list,
    const char         *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clBuildProgram_ptr;
    if(func != nullptr)
    {
        return func(program, num_devices, device_list, options, pfn_notify, user_data);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clEnqueueNDRangeKernel(
    cl_command_queue command_queue,
    cl_kernel        kernel,
    cl_uint          work_dim,
    const size_t    *global_work_offset,
    const size_t    *global_work_size,
    const size_t    *local_work_size,
    cl_uint          num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event        *event)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clEnqueueNDRangeKernel_ptr;
    if(func != nullptr)
    {
        return func(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clSetKernelArg(
    cl_kernel   kernel,
    cl_uint     arg_index,
    size_t      arg_size,
    const void *arg_value)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clSetKernelArg_ptr;
    if(func != nullptr)
    {
        return func(kernel, arg_index, arg_size, arg_value);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clRetainMemObject(cl_mem memobj)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clRetainMemObject_ptr;
    if(func != nullptr)
    {
        return func(memobj);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clReleaseMemObject(cl_mem memobj)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clReleaseMemObject_ptr;
    if(func != nullptr)
    {
        return func(memobj);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clEnqueueUnmapMemObject(
    cl_command_queue command_queue,
    cl_mem           memobj,
    void            *mapped_ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event        *event)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clEnqueueUnmapMemObject_ptr;
    if(func != nullptr)
    {
        return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clRetainCommandQueue(cl_command_queue command_queue)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clRetainCommandQueue_ptr;
    if(func != nullptr)
    {
        return func(command_queue);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clReleaseContext(cl_context context)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clReleaseContext_ptr;
    if(func != nullptr)
    {
        return func(context);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}
cl_int clReleaseEvent(cl_event event)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clReleaseEvent_ptr;
    if(func != nullptr)
    {
        return func(event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clEnqueueWriteBuffer(
    cl_command_queue command_queue,
    cl_mem           buffer,
    cl_bool          blocking_write,
    size_t           offset,
    size_t           size,
    const void      *ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event        *event)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clEnqueueWriteBuffer_ptr;
    if(func != nullptr)
    {
        return func(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clEnqueueReadBuffer(
    cl_command_queue command_queue,
    cl_mem           buffer,
    cl_bool          blocking_read,
    size_t           offset,
    size_t           size,
    void            *ptr,
    cl_uint          num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event        *event)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clEnqueueReadBuffer_ptr;
    if(func != nullptr)
    {
        return func(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetProgramBuildInfo(
    cl_program            program,
    cl_device_id          device,
    cl_program_build_info param_name,
    size_t                param_value_size,
    void                 *param_value,
    size_t               *param_value_size_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetProgramBuildInfo_ptr;
    if(func != nullptr)
    {
        return func(program, device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clRetainProgram(cl_program program)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clRetainProgram_ptr;
    if(func != nullptr)
    {
        return func(program);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

void *clEnqueueMapBuffer(
    cl_command_queue command_queue,
    cl_mem           buffer,
    cl_bool          blocking_map,
    cl_map_flags     map_flags,
    size_t           offset,
    size_t           size,
    cl_uint          num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event        *event,
    cl_int          *errcode_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clEnqueueMapBuffer_ptr;
    if(func != nullptr)
    {
        return func(command_queue, buffer, blocking_map, map_flags, offset, size, num_events_in_wait_list, event_wait_list, event, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clReleaseCommandQueue_ptr;
    if(func != nullptr)
    {
        return func(command_queue);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_program clCreateProgramWithBinary(
    cl_context            context,
    cl_uint               num_devices,
    const cl_device_id   *device_list,
    const size_t         *lengths,
    const unsigned char **binaries,
    cl_int               *binary_status,
    cl_int               *errcode_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clCreateProgramWithBinary_ptr;
    if(func != nullptr)
    {
        return func(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}

cl_int clRetainContext(cl_context context)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clRetainContext_ptr;
    if(func != nullptr)
    {
        return func(context);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clReleaseProgram(cl_program program)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clReleaseProgram_ptr;
    if(func != nullptr)
    {
        return func(program);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clFlush(cl_command_queue command_queue)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clFlush_ptr;
    if(func != nullptr)
    {
        return func(command_queue);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clFinish(cl_command_queue command_queue)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clFinish_ptr;
    if(func != nullptr)
    {
        return func(command_queue);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetProgramInfo(
    cl_program      program,
    cl_program_info param_name,
    size_t          param_value_size,
    void           *param_value,
    size_t         *param_value_size_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetProgramInfo_ptr;
    if(func != nullptr)
    {
        return func(program, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_kernel clCreateKernel(
    cl_program  program,
    const char *kernel_name,
    cl_int     *errcode_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clCreateKernel_ptr;
    if(func != nullptr)
    {
        return func(program, kernel_name, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}

cl_int clRetainKernel(cl_kernel kernel)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clRetainKernel_ptr;
    if(func != nullptr)
    {
        return func(kernel);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_mem clCreateBuffer(
    cl_context   context,
    cl_mem_flags flags,
    size_t       size,
    void        *host_ptr,
    cl_int      *errcode_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clCreateBuffer_ptr;
    if(func != nullptr)
    {
        return func(context, flags, size, host_ptr, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}

cl_program clCreateProgramWithSource(
    cl_context    context,
    cl_uint       count,
    const char **strings,
    const size_t *lengths,
    cl_int       *errcode_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clCreateProgramWithSource_ptr;
    if(func != nullptr)
    {
        return func(context, count, strings, lengths, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}

cl_int clReleaseKernel(cl_kernel kernel)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clReleaseKernel_ptr;
    if(func != nullptr)
    {
        return func(kernel);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetDeviceIDs(cl_platform_id platform,
                      cl_device_type device_type,
                      cl_uint        num_entries,
                      cl_device_id *devices,
                      cl_uint       *num_devices)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetDeviceIDs_ptr;
    if(func != nullptr)
    {
        return func(platform, device_type, num_entries, devices, num_devices);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetDeviceInfo(cl_device_id   device,
                       cl_device_info param_name,
                       size_t         param_value_size,
                       void          *param_value,
                       size_t        *param_value_size_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetDeviceInfo_ptr;
    if(func != nullptr)
    {
        return func(device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetMemObjectInfo(cl_mem      memobj,
                          cl_mem_info param_name,
                          size_t      param_value_size,
                          void       *param_value,
                          size_t     *param_value_size_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetMemObjectInfo_ptr;
    if(func != nullptr)
    {
        return func(memobj, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clRetainEvent(cl_event event)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clRetainEvent_ptr;
    if(func != nullptr)
    {
        return func(event);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetPlatformIDs_ptr;
    if(func != nullptr)
    {
        return func(num_entries, platforms, num_platforms);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int
clGetKernelWorkGroupInfo(cl_kernel                 kernel,
                         cl_device_id              device,
                         cl_kernel_work_group_info param_name,
                         size_t                    param_value_size,
                         void                     *param_value,
                         size_t                   *param_value_size_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetKernelWorkGroupInfo_ptr;
    if(func != nullptr)
    {
        return func(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int
clGetCommandQueueInfo(cl_command_queue      command_queue,
                      cl_command_queue_info param_name,
                      size_t                param_value_size,
                      void                 *param_value,
                      size_t               *param_value_size_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetCommandQueueInfo_ptr;
    if(func != nullptr)
    {
        return func(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int
clGetKernelInfo(cl_kernel      kernel,
                cl_kernel_info param_name,
                size_t         param_value_size,
                void          *param_value,
                size_t        *param_value_size_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetKernelInfo_ptr;
    if(func != nullptr)
    {
        return func(kernel, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_int
clGetEventProfilingInfo(cl_event          event,
                        cl_profiling_info param_name,
                        size_t            param_value_size,
                        void             *param_value,
                        size_t           *param_value_size_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clGetEventProfilingInfo_ptr;
    if(func != nullptr)
    {
        return func(event, param_name, param_value_size, param_value, param_value_size_ret);
    }
    else
    {
        return CL_OUT_OF_RESOURCES;
    }
}

cl_mem
clImportMemoryARM(cl_context                      context,
                  cl_mem_flags                    flags,
                  const cl_import_properties_arm *properties,
                  void                           *memory,
                  size_t                          size,
                  cl_int                         *errcode_ret)
{
    CLSymbols::get().load_default();
    auto func = CLSymbols::get().clImportMemoryARM_ptr;
    if(func != nullptr)
    {
        return func(context, flags, properties, memory, size, errcode_ret);
    }
    else
    {
        if(errcode_ret != nullptr)
        {
            *errcode_ret = CL_OUT_OF_RESOURCES;
        }
        return nullptr;
    }
}