#ifndef CHAINFK_H
#define CHAINFK_H
#include "Msnhnet/robot/MsnhChain.h"

namespace Msnhnet
{

class MsnhNet_API ChainFK
{
public:
    static Frame jointToCartesian(const Chain &chain, const std::vector<double> joints, int segNum = -1);
};

}

#endif 

