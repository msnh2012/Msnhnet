#ifndef MSNHCLSCHEDULER_H
#define MSNHCLSCHEDULER_H

#include "CL\cl.h"
#include <iostream>
#include <fstream>
#include <map>
#include <mutex>

namespace Msnhnet
{

const std::map<int, std::string> OpenCLProgramMap = {
    {0, "H:\\projects\\Msnhnet_restart\\src\\layers\\opencl\\cl\\convolution.cl"},
    {1, ""},
    {2, ""},
    {3, "H:\\projects\\Msnhnet_restart\\src\\layers\\opencl\\cl\\maxPool.cl"},
    {4, ""},
    {5, ""},
    {6, ""},
    {7, ""},
    {8, ""},
    {9, ""},
    {10, ""},
    {11, ""},
    {12, "H:\\projects\\Msnhnet_restart\\src\\layers\\opencl\\cl\\activations.cl"},
    {13, "H:\\projects\\Msnhnet_restart\\src\\layers\\opencl\\cl\\batchNorm.cl"},
    {14, ""},
    {15, ""},
    {16, ""},
    {17, ""},
    {18, ""},
    {19, ""},
    {20, ""},
    {21, ""},
    {22, ""},
    {23, ""},
    {24, ""},
    {25, ""},
    {26, ""},
    {27, ""},
    {28, ""},
    {29, ""},
    {30, ""},
    {31, "H:\\projects\\Msnhnet_restart\\src\\layers\\opencl\\cl\\padding.cl"},
    {32, ""},
    {33, ""}
};

class clScheduler 
{
public:
    clScheduler();
    ~clScheduler();
    clScheduler(const clScheduler &) = delete;
    clScheduler &operator=(const clScheduler &) = delete; 

    static clScheduler& get();
    void init();

    cl_context& context();
    cl_command_queue& queue();
    cl_device_id* device();

    cl_kernel buildKernel(const int &layerType, const std::string &kernelName);

private:

    cl_int getPlatform();
    cl_int getCLDeviceId();

    //static std::once_flag _initialize_symbols;
    
    cl_platform_id      _platform;
    cl_context          _context;
    cl_command_queue    _queue;
    cl_device_id*       _devices = nullptr;

    std::map<int, cl_program> programMap;
    // std::map<std::pair<std::string, std::string>, cl_program> programMap;

};
}

#endif