#ifdef USE_X86
# include "Msnhnet/layers/MsnhActivationsAvx.h"
namespace Msnhnet
{
const __m256 ActivationsAvx::c1f     = _mm256_set1_ps(1.f);
const __m256 ActivationsAvx::c0f     = _mm256_set1_ps(0);
const __m256 ActivationsAvx::c2f     = _mm256_set1_ps(2.f);
const __m256 ActivationsAvx::c0_1f   = _mm256_set1_ps(0.1f);
const __m256 ActivationsAvx::c0_5f   = _mm256_set1_ps(0.5f);
const __m256 ActivationsAvx::c0_01f  = _mm256_set1_ps(0.01f);
const __m256 ActivationsAvx::c4f     = _mm256_set1_ps(4.f);
const __m256 ActivationsAvx::cN4f    = _mm256_set1_ps(-4.f);
const __m256 ActivationsAvx::c0_001f = _mm256_set1_ps(0.001f);
const __m256 ActivationsAvx::c0_125f = _mm256_set1_ps(0.125f);
const __m256 ActivationsAvx::c3f     = _mm256_set1_ps(3.f);
const __m256 ActivationsAvx::c0_16f  = _mm256_set1_ps(0.16666667f);

void ActivationsAvx::activateAvx8( float* const &x, const ActivationType &actType, const float &params)
{
    switch (actType)
    {
    case LOGISTIC:
        logisticActivateSize8(x);
        break;
    case LOGGY:
        loggyActivateSize8(x);
        break;
    case RELU:
        reluActivateSize8(x);
        break;
    case RELU6:
        relu6ActivateSize8(x);
        break;
    case ELU:
        eluActivateSize8(x);
        break;
    case SELU:
        seluActivateSize8(x);
        break;
    case RELIE:
        relieActivateSize8(x);
        break;
    case RAMP:
        rampActivateSize8(x);
        break;
    case LEAKY:
        leakyActivateSize8(x,params);
        break;
    case TANH:
        tanhActivateSize8(x);
        break;
    case PLSE:
        plseActivateSize8(x);
        break;
    case STAIR:
        stairActivateSize8(x);
        break;
    case HARDTAN:
        hardtanActivateSize8(x);
        break;
    case LHTAN:
        lhtanActivateSize8(x);
        break;
    case SOFT_PLUS:
        softplusActivateSize8(x,params);
        break;
    case MISH:
        mishActivateSize8(x);
        break;
    case SWISH:
        swishActivateSize8(x);
        break;
    case HARD_SWISH:
        hardSwishActivateSize8(x);
        break;
    }
}

}
#endif

