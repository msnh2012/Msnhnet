#include "Msnhnet/layers/MsnhActivationsNeon.h"
#ifdef USE_NEON
namespace Msnhnet
{
void ActivationsNeon::activateNeon4(float * const &x, const ActivationType &actType, const float &params)
{
    switch (actType)
    {
    case LOGISTIC:
        logisticActivateSize4(x);
        break;
    case LOGGY:
        loggyActivateSize4(x);
        break;
    case RELU:
        reluActivateSize4(x);
        break;
    case RELU6:
        relu6ActivateSize4(x);
        break;
    case ELU:
        eluActivateSize4(x);
        break;
    case SELU:
        seluActivateSize4(x);
        break;
    case RELIE:
        relieActivateSize4(x);
        break;
    case RAMP:
        rampActivateSize4(x);
        break;
    case LEAKY:
        leakyActivateSize4(x,params);
        break;
    case TANH:
        tanhActivateSize4(x);
        break;
    case PLSE:
        plseActivateSize4(x);
        break;
    case STAIR:
        stairActivateSize4(x);
        break;
    case HARDTAN:
        hardtanActivateSize4(x);
        break;
    case LHTAN:
        lhtanActivateSize4(x);
        break;
    case SOFT_PLUS:
        softplusActivateSize4(x,params);
        break;
    case MISH:
        mishActivateSize4(x);
        break;
    case SWISH:
        swishActivateSize4(x);
        break;
    }
}
}
#endif
