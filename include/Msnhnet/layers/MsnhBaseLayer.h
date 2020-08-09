#ifndef MSNHBASELAYER_H
#define MSNHBASELAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/core/MsnhSimd.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/utils/MsnhTimeUtil.h"

#ifdef USE_GPU
#include "Msnhnet/config/MsnhnetCuda.h"
#endif

namespace Msnhnet
{
class NetworkState;
class MsnhNet_API BaseLayer
{
public:
    BaseLayer();
    virtual ~BaseLayer();

    static bool     supportAvx;
    static bool     supportFma;
    static bool     isPreviewMode;
    static bool     onlyUseCuda;
    static bool     useFp16;

#ifdef USE_GPU
    static cudaEvent_t     _start;
    static cudaEvent_t     _stop;
#endif

    static void setPreviewMode(const bool &isPreviewMode);

    static void setForceUseCuda(const bool &forceUseCuda);

    static void setUseFp16(const bool &useFp16);

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
    float *getGpuOutput() const;
    void recordCudaStart();
    void recordCudaStop();
#endif
    virtual void loadAllWeigths(std::vector<float> &weights);

    static void initSimd();
    inline void releaseArr(void * value)
    {
        if(value!=nullptr)
        {
            delete[] value;
            value = nullptr;
        }
    }

    LayerType type() const;

    ActivationType activation() const;

    int getOutHeight() const;

    int getOutWidth() const;

    int getOutChannel() const;

    int getOutputNum() const;

    void setOutHeight(int getOutHeight);

    void setOutWidth(int getOutWidth);

    void setOutChannel(int getOutChannel);

    float *getOutput() const;

    int getInputNum() const;

    size_t getWorkSpaceSize() const;

    void setWorkSpaceSize(const size_t &getWorkSpaceSize);

    size_t getNumWeights() const;

    std::string getLayerDetail() const;

    int getHeight() const;

    int getWidth() const;

    int getChannel() const;

    float getForwardTime() const;

    std::string getLayerName() const;

    int getBatch() const;

    ActivationType getActivation() const;

    size_t getInputSpaceSize() const;

protected:
    LayerType          _type;                       

    ActivationType     _activation;                 

    std::vector<float> _actParams;

    int             _num             =  0;       

    size_t          _workSpaceSize   =  0;
    size_t          _inputSpaceSize  =  0;

    int             _height          =  0;
    int             _width           =  0;
    int             _channel         =  0;

    int             _outHeight       =  0;
    int             _outWidth        =  0;
    int             _outChannel      =  0;

    int             _inputNum        =  0;
    int             _outputNum       =  0;

    size_t          _numWeights      =  0;       

    int             _batch           =  0;
    float          *_output          =  nullptr; 

    float           _bFlops          =  0;

#ifdef USE_GPU
    float          *_gpuOutput       =  nullptr;
#endif

    std::string     _layerName       =  "BaseLayer";
    std::string     _layerDetail     =  "";

    float           _forwardTime     =  0;
};
}

#endif 

