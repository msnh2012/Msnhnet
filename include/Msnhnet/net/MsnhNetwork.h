#ifndef MSNHNETWORK_H
#define MSNHNETWORK_H
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class BaseLayer;
class NetworkState;
class Network
{
public:
    Network(){}
    ~Network(){}

    std::vector<BaseLayer*>   layers;

    int             batch           =   0;       

    int             height          =   0;
    int             width           =   0;
    int             channels        =   0;
    int             inputNum        =   0;

    float          *input           =  nullptr; 

    float          *output          =  nullptr; 

    void forward(NetworkState &netState);
};

class MsnhNet_API NetworkState
{
public:

    float           *input          =  nullptr; 

    int             inputNum        =  0;
    Network         *net            =  nullptr;
    int             fixNan          =  0;

    float           *workspace      =  nullptr; 

    float           *memPool2       =  nullptr;
    float           *memPool1       =  nullptr;
#ifdef USE_GPU
    float           *gpuWorkspace   =  nullptr; 

    float           *gpuMemPool1    =  nullptr;
    float           *gpuMemPool2    =  nullptr;

    float           *gpuInputFp16   =  nullptr;
#endif

    template<typename T>
    inline void releaseArr(T *& value)
    {

        MemoryManager::effcientDelete<T>(value);
    }

    ~NetworkState();

    inline void shuffleInOut()
    {
        uint8_t temp        = _inputWorkSpace;
        _inputWorkSpace     = _outputWorkSpace;
        _outputWorkSpace    = temp;
    }

    uint8_t getInputWorkSpace() const;

    uint8_t getOutputWorkSpace() const;

    float *getInput() const;

    float *getOutput() const;

#ifdef USE_GPU
    float *getGpuInput() const;

    float *getGpuOutput() const;

    uint8_t getGpuInputWorkSpace() const;

    uint8_t getGpuOutputWorkSpace() const;

    inline void shuffleGpuInOut()
    {
        uint8_t temp            = _gpuInputWorkSpace;
        _gpuInputWorkSpace      = _gpuOutputWorkSpace;
        _gpuOutputWorkSpace     = temp;
    }
#endif

private:
    uint8_t _inputWorkSpace   = 0;
    uint8_t _outputWorkSpace  = 1;
#ifdef USE_GPU
    uint8_t _gpuInputWorkSpace   = 0;
    uint8_t _gpuOutputWorkSpace  = 1;
#endif

};
}
#endif 

