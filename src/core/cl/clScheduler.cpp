#include "Msnhnet/core/cl/clScheduler.h"

namespace Msnhnet
{

int convertToString(const char *filename, std::string& s){
    size_t size;
    char* str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));
    if (f.is_open()){
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size+1];
        if (!str) {
            f.close();
            return 0;
        }
        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    std::cout << "Error: failed to open file\n" << filename << std::endl;
    return -1;
}

clScheduler::clScheduler()
    : _context(), _queue(), _platform(), _devices()
{
    init();
}

clScheduler::~clScheduler(){
    // TODO
}

clScheduler& clScheduler::get(){
    //std::call_once(_initialize_symbols);
    static clScheduler scheduler;
    return scheduler;
}


cl_context& clScheduler::context(){
    return _context;
}

cl_command_queue& clScheduler::queue(){
    return _queue;
}

cl_int clScheduler::getPlatform(){

    cl_uint numPlatforms;

    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS){
        std::cout << "Error: Getting Platforms failed!" << std::endl;
        return status;
    }

    std::cout << numPlatforms << " platform available." << std::endl;

    if (numPlatforms > 0){
        cl_platform_id* platforms = (cl_platform_id* ) malloc (numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        _platform = platforms[0];
        free(platforms);
        return CL_SUCCESS;
    }
    else{
        return status;
    }
}

cl_int clScheduler::getCLDeviceId() {
    cl_uint numDevices = 0;
    cl_int status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    // std::cout << numDevices << " gpu device available" << std::endl;
    if (status != CL_SUCCESS){
        return status;
    }

    if (numDevices > 0) {
        _devices = (cl_device_id*) malloc (numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, numDevices, _devices, NULL);
        return status;
    }
    return -1;
}


void clScheduler::init(){

    cl_int status = getPlatform();
    if(status != CL_SUCCESS){
        std::cout << "get platform faild, status : " << status << std::endl;
        return;
    }

    status = getCLDeviceId();    
    if (status != CL_SUCCESS){
        std::cout << "Error: Getting Devices failed!" << std::endl;
        return;
    }

    _context = clCreateContext(NULL, 1, _devices, NULL, NULL, NULL);
    _queue = clCreateCommandQueue(_context, _devices[0], 0, NULL);
    

}

cl_kernel clScheduler::buildKernel(const int &layerType, const std::string &kernelName){


    cl_program program;
    cl_int status;
    auto programInter = programMap.find(layerType);
    if (programInter != programMap.end()){
        program = programInter->second;
    } else {
        auto clFile = OpenCLProgramMap.find(layerType);
        std::string sourceStr;
        status = convertToString(clFile->second.c_str(), sourceStr);
        const char* source = sourceStr.c_str();
        size_t sourceSize[] = {strlen(source)};
        program = clCreateProgramWithSource(_context, 1, &source, sourceSize, &status);
        std::cout << "create program with source " << status << std::endl;

        status |= clBuildProgram(program, 1, _devices, NULL, NULL, NULL);
        if (status != CL_SUCCESS){
            std::cout << "Error: Build Program failed!" << std::endl;
            return NULL;
        }
        std::cout << "openCL init finish" << std::endl;
        programMap.emplace(layerType, program);
    }
    cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &status);
    if (status != CL_SUCCESS)
    {
        std::cout << "Create " << kernelName << " kernel failed" << std::endl;
    }
    
    return kernel;
}




}