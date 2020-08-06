#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{
int  Cuda::blockThread  = 512;

cudaStream_t Cuda::stream;
bool Cuda::streamInited = false;

cublasHandle_t Cuda::handle;
bool Cuda::handleInited = false;

void Cuda::cudaCheck(cudaError_t status, const char *fun, const char *file, const int &line)
{
    cudaError_t lastStatus = cudaGetLastError();

    if(status != cudaSuccess)
    {
        const char* str = cudaGetErrorString(status);

        std::string err;

        err = "Cuda Error: " + std::string(str) + "\n";

        std::cerr<<err<<" File:  "<<file<<" : "<<line<<"  fun:  "<<fun<< std::endl;

        exit(EXIT_FAILURE);
    }

    if(lastStatus != cudaSuccess)
    {
        const char* str = cudaGetErrorString(lastStatus);

        std::string err;

        err = "Cuda Prev Error: " + std::string(str) + "\n";

        std::cerr<<err<<" File:  "<<file<<" : "<<line<<"  fun:  "<<fun<< std::endl;

        exit(EXIT_FAILURE);
    }
}

void Cuda::cublasCheck(cublasStatus_t status,  const char *fun, const char *file, const int &line)
{
    if(status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr<<"Error code: "<<status<<" \nFile:  "<<file<<" : "<<line<<"  fun:  "<<fun<< std::endl;
        exit(EXIT_FAILURE);
    }
}

int Cuda::getDevice()
{
    int n = 0;
    CUDA_CHECK(cudaGetDevice(&n));
    return n;
}

void Cuda::setBestGPU()
{
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));

    if(numDevices > 1)
    {
        int maxMultProcessors = 0;
        int maxDevice         = 0;

        for (int device = 0; device < numDevices; ++device)
        {
            cudaDeviceProp props;
            CUDA_CHECK(cudaGetDeviceProperties(&props, device));

            if(maxMultProcessors < props.multiProcessorCount)
            {
                maxMultProcessors = props.multiProcessorCount;
                maxDevice = device;
            }
        }
        CUDA_CHECK(cudaSetDevice(maxDevice));
    }
}

std::string Cuda::getDeviceInfo()
{
    int deviceCnt = 0;

    CUDA_CHECK(cudaGetDeviceCount(&deviceCnt));

    if(deviceCnt == 0)
    {
        throw Exception(1, "There are no available device(s) that support CUDA", __FILE__, __LINE__, __FUNCTION__);
    }

    std::string msg;

    msg  = msg + "Device  nums: " + std::to_string(deviceCnt) + "\n";

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    msg  = msg + "Device0 name: " + deviceProp.name + "\n";

    int cudaDriver  = 0;
    int cudaRuntime = 0;

    CUDA_CHECK(cudaDriverGetVersion(&cudaDriver));
    CUDA_CHECK(cudaRuntimeGetVersion(&cudaRuntime));

    if(cudaDriver/1000<5)
    {
        throw Exception(1, "Cuda version must > 5.0", __FILE__, __LINE__, __FUNCTION__);
    }

    msg  = msg + "Cuda driver : " + std::to_string(cudaDriver/1000) + "." + std::to_string((cudaDriver%100)/10) + "\n";
    msg  = msg + "Cuda runtime: " + std::to_string(cudaRuntime/1000) + "." + std::to_string((cudaRuntime%100)/10) + "\n";
    return msg;
}

dim3 Cuda::getGrid(const size_t &n)
{

    size_t k = (n-1)/blockThread + 1; 

    int x = 1;
    int y = 1;

    if(k > 65536) 

    {
        x = static_cast<int>(ceil(sqrt(k)));
        y=  static_cast<int>((n-1)/(x*blockThread) + 1);
    }
    else
    {
        x =  static_cast<int>(k);
    }
    return dim3(x,y,1);
}

cudaStream_t Cuda::getCudaStream()
{
    if(!streamInited)
    {
        cudaError_t status      =  cudaStreamCreate(&stream);

        if (status != cudaSuccess)
        {
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));
        }

        streamInited = true;
    }
    return stream;
}

cublasHandle_t Cuda::getBlasHandle()
{
    if(!handleInited)
    {
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, getCudaStream()));
        handleInited = true;
    }
    return handle;
}

void Cuda::deleteBlasHandle()
{
   CUBLAS_CHECK(cublasDestroy(handle));
}

float *Cuda::makeCudaArray(float * const &x, const size_t &n, const cudaMemcpyKind &copyType)
{
    float *gpuX;
    CUDA_CHECK(cudaMalloc((void **)&gpuX, n*sizeof(float)));
    if(x != nullptr)
    {
        CUDA_CHECK(cudaMemcpyAsync(gpuX, x, n*sizeof(float), copyType, getCudaStream()));
    }
    if(gpuX == nullptr)
    {
        throw Exception(1, "Cuda malloc failed. \n",__FILE__,__LINE__, __FUNCTION__);
    }
    return gpuX;
}

void Cuda::pushCudaArray(float * const &gpuX, float * const &x, const size_t &n)
{
    CUDA_CHECK(cudaMemcpyAsync(gpuX, x, n*sizeof(float), cudaMemcpyHostToDevice, getCudaStream()));
}

void Cuda::pullCudaArray(float * const &gpuX, float * const &x, const size_t &n)
{
    CUDA_CHECK(cudaMemcpyAsync(x, gpuX , n*sizeof(float), cudaMemcpyDeviceToHost, getCudaStream()));
    cudaStreamSynchronize(getCudaStream());
}

void Cuda::freeCuda(float * const &gpuX)
{
    if(gpuX != nullptr)
    {
       CUDA_CHECK(cudaFree(gpuX));
    }

}

}

