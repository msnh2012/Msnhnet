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

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

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
    static float *mallocCudaArray(const size_t &n);

    static __half *makeFp16ArrayFromFp32(float * const &x, const size_t &n);

    static void  pushCudaArray(float *const &gpuX, float *const &x, const size_t &n);

    static void  pullCudaArray(float *const &gpuX, float *const &x, const size_t &n);
    static void  freeCuda(float *const &gpuX);

    static void  fp32ToFp16(float *const &fp32, const size_t &size, float * const &fp16);
    static void  fp16ToFp32(float * const &fp16, const size_t &size, float *const &fp32);

#ifdef USE_CUDNN
    static cudnnHandle_t cudnnHandle;
    static bool cudnnHandleInited;

    static void cudnnCheck(cudnnStatus_t status, const char *fun, const char* file, const int &line);
#define CUDNN_CHECK(X) Msnhnet::Cuda::cudnnCheck(X,__FUNCTION__, __FILE__, __LINE__)

    static cudnnHandle_t getCudnnHandle();
#endif

};
}

#endif 

