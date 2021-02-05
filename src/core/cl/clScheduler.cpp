#include "Msnhnet/core/clScheduler.cpp"

clScheduler& clScheduler::get(){
    static clScheduler scheduler;
    return scheduler;
}

void clScheduler::init(){
    cl_int errNum;

    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        return NULL;
    }

    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    
    cl_context_properties contextProperties[] =
        {
            CL_CONTEXT_PLATFORM, 
            (cl_context_properties) firstPlatformId,
            0
        };
    _context = clCreateContext(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS) {
        //
    }

    cl_device_id* devices;
    

}