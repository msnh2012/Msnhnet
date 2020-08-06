#ifndef MSNHNETCUDA_H
#define MSNHNETCUDA_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <math.h>

namespace Msnhnet
{
class Cuda
{
public:
    static cudaStream_t stream;
    static bool streamInited;

    static cublasHandle_t handle;
    static bool handleInited;

    static int blockThread;
    static void cudaCheck(cudaError_t status, const char *fun, const char* file, const int &line);
#define CUDA_CHECK(X) Msnhnet::Cuda::cudaCheck(X,__FUNCTION__,__FILE__, __LINE__)
    static void cublasCheck(cublasStatus_t status, const char *fun, const char* file, const int &line);
#define CUBLAS_CHECK(X) Msnhnet::Cuda::cublasCheck(X,__FUNCTION__, __FILE__, __LINE__)
    static int getDevice();
    static void setBestGPU();
    static std::string getDeviceInfo();
    static dim3 getGrid(const size_t &n);
    static cudaStream_t getCudaStream();
    static cublasHandle_t getBlasHandle();
    static void deleteBlasHandle();
    static float *makeCudaArray(float *const &x, const size_t &n, const cudaMemcpyKind &copyType=cudaMemcpyHostToDevice);

    static void  pushCudaArray(float *const &gpuX, float *const &x, const size_t &n);

    static void  pullCudaArray(float *const &gpuX, float *const &x, const size_t &n);

    static void freeCuda(float *const &gpuX);

};
}

#endif 

