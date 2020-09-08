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
    VariableOpLayer(const int &batch, const int &width, const int &height, const int &channel, std::vector<int> &getInputLayerIndexes, const VariableOpParams::VarOpType &getVarOpType, const float &getConstVal);

    virtual void forward(NetworkState &netState);

    virtual void mallocMemory();

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    std::vector<int> getInputLayerIndexes() const;
    float getConstVal() const;
    VariableOpParams::VarOpType getVarOpType() const;

private:
    std::vector<int> _inputLayerIndexes;
    VariableOpParams::VarOpType _varOpType;
    float            _constVal;
};

}

#endif 

