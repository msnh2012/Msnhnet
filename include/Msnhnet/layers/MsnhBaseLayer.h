#ifndef MSNHBASELAYER_H
#define MSNHBASELAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/core/MsnhSimd.h"
#include "Msnhnet/utils/MsnhExport.h"

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

    static void setPreviewMode(const bool &isPreviewMode);

    virtual void forward(NetworkState &netState);
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

protected:
    LayerType          _type;                       

    ActivationType     _activation;                 

    std::vector<float> _actParams;

    int             _num             =  0;       

    size_t          _workSpaceSize   =  0;

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

    std::string     _layerName       =  "BaseLayer";
    std::string     _layerDetail     =  "";

    float           _forwardTime     =  0;
};
}

#endif 

