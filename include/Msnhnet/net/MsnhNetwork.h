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

    float           *truth          =  nullptr; 
    float           *input          =  nullptr; 
    int             inputNum        =  0;

    Network         *net            =  nullptr;

    float           *workspace      =  nullptr; 
    inline void releaseArr(float * value)
    {
        if(value!=nullptr)
        {
            delete[] value;
            value = nullptr;
        }
    }
    ~NetworkState();
};
}
#endif 
