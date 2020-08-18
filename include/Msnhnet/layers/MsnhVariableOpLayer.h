#ifndef MSNHVARIABLEOPLAYER_H
#define MSNHVARIABLEOPLAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/io/MsnhParser.h"

namespace Msnhnet
{

class MsnhNet_API VariableOpLayer : public BaseLayer
{
public:
    VariableOpLayer(const int &batch, std::vector<int> &inputLayerIndexes, std::vector<int> &inputLayerOutputs, const VariableOpParams::VarOpType &varOpType, const float &constVal);

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    std::vector<int> inputLayerIndexes() const;
    std::vector<int> inputLayerOutputs() const;
    float constVal() const;
    VariableOpParams::VarOpType varOpType() const;

private:
    std::vector<int> _inputLayerIndexes;
    std::vector<int> _inputLayerOutputs;
    VariableOpParams::VarOpType _varOpType;
    float            _constVal;
};

}

#endif 

