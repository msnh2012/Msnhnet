#include "Msnhnet/layers/MsnhActivations.h"
namespace Msnhnet
{
ActivationType Activations::getActivation(const std::string &msg)
{
    if (msg=="logistic") return LOGISTIC;
    if (msg=="swish") return SWISH;
    if (msg=="mish") return MISH;
    if (msg=="normalize_channels") return NORM_CHAN;
    if (msg=="normalize_channels_softmax") return NORM_CHAN_SOFTMAX;
    if (msg=="normalize_channels_softmax_maxval") return NORM_CHAN_SOFTMAX_MAXVAL;
    if (msg=="loggy") return LOGGY;
    if (msg=="relu") return RELU;
    if (msg=="relu6") return RELU6;
    if (msg=="elu") return ELU;
    if (msg=="selu") return SELU;
    if (msg=="relie") return RELIE;
    if (msg=="plse") return PLSE;
    if (msg=="hardtan") return HARDTAN;
    if (msg=="lhtan") return LHTAN;
    if (msg=="ramp") return RAMP;
    if (msg=="leaky") return LEAKY;
    if (msg=="tanh") return TANH;
    if (msg=="softplus") return SOFT_PLUS;
    if (msg=="stair") return STAIR;
    if (msg=="none") return NONE;
    return RELU;
}

string Activations::getActivationStr(const ActivationType &type)
{
    if (type==LOGISTIC) return "logistic";
    if (type==SWISH) return "swish";
    if (type==MISH) return "mish";
    if (type==NORM_CHAN) return "nc";
    if (type==NORM_CHAN_SOFTMAX) return "ncs";
    if (type==NORM_CHAN_SOFTMAX_MAXVAL) return "ncsm";
    if (type==LOGGY) return "loggy";
    if (type==RELU) return "relu";
    if (type==RELU6) return "relu6";
    if (type==ELU) return "elu";
    if (type==SELU) return "selu";
    if (type==RELIE) return "relie";
    if (type==PLSE) return "plse";
    if (type==HARDTAN) return "hardtan";
    if (type==LHTAN) return "lhtan";
    if (type==RAMP) return "ramp";
    if (type==LEAKY) return "leaky";
    if (type==TANH) return "tanh";
    if (type==SOFT_PLUS) return "softplus";
    if (type==STAIR) return "stair";
    if (type==NONE) return "none";
    return "unknow";
}
float Activations::activate(const float &x, const ActivationType &actType, const float &params)
{
    switch (actType)
    {
    case LOGISTIC:
        return logisticActivate(x);
    case LOGGY:
        return loggyActivate(x);
    case RELU:
        return reluActivate(x);
    case RELU6:
        return relu6Activate(x);
    case ELU:
        return eluActivate(x);
    case SELU:
        return seluActivate(x);
    case RELIE:
        return relieActivate(x);
    case RAMP:
        return rampActivate(x);
    case LEAKY:
        return leakyActivate(x, params);
    case TANH:
        return tanhActivate(x);
    case PLSE:
        return plseActivate(x);
    case STAIR:
        return stairActivate(x);
    case HARDTAN:
        return hardtanActivate(x);
    case LHTAN:
        return lhtanActivate(x);
    case SOFT_PLUS:
        return softplusActivate(x, params);
    case MISH:
        return mishActivate(x);
    case SWISH:
        return swishActivate(x);
    case NONE:
        return x;
    default:
        return 0;
    }
}

void Activations::activateArrayNormCh(float * const &x, const int &numX, const int &batch, const int &channels, const int &whStep, float * const &output)
{
    /* TODO: */
    (void) batch;
    int size = numX/channels;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < size; ++i)
    {
        int whIndex = i % whStep;
        int b       = i / whStep;

        const float eps = 0.0001f;
        if(i<size)
        {
            float sum = eps;

            for (int k = 0; k < channels; ++k)
            {
                float val = x[whIndex + k*whStep + b*whStep*channels];
                if(val > 0)
                {
                    sum+=val;
                }
            }

            for (int k = 0; k < channels; ++k)
            {
                float val = x[whIndex + k*whStep + b*whStep*channels];
                if(val >0)
                {
                    val = val/sum;
                }
                else
                {
                    val = 0;
                }
                output[whIndex + k*whStep + b*whStep*channels] = val;
            }
        }
    }

}

void Activations::activateArrayNormChSoftMax(float * const &x, const int &numX, const int &batch, const int &channels, const int &whStep, float * const &output, const int &useMaxVal)
{
    /* TODO: */
    (void) batch;
    int size = numX/channels;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < size; ++i)
    {
        int whIndex = i % whStep;
        int b       = i / whStep;

        const float eps = 0.0001f;
        if(i<size)
        {
            float sum    = eps;
            float maxVal = -FLT_MAX;

            if(useMaxVal)
            {
                for (int k = 0; k < channels; ++k)
                {
                    float val = x[whIndex + k*whStep + b*whStep*channels];
                    if(val > maxVal)
                    {
                        maxVal = val;
                    }
                }
            }
            else
            {
                maxVal = 0;
            }

            for (int k = 0; k < channels; ++k)
            {
                float val = x[whIndex + k*whStep + b*whStep*channels];
                sum += expf(val - maxVal);
            }

            for (int k = 0; k < channels; ++k)
            {
                float val = x[whIndex + k*whStep + b*whStep*channels];
                val = expf(val - maxVal) / sum;
                output[whIndex + k*whStep + b*whStep*channels] = val;
            }
        }
    }
}

void Activations::activateArray(float *const &x, const int &numX, const ActivationType &actType, const bool &useAVX, const float &param)
{

#ifdef USE_X86
    if(useAVX)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<numX/8;++i)
        {
            ActivationsAvx::activateAvx8(x+i*8,actType,param);
        }

        for(int i=(numX/8)*8 ; i<numX;++i)
        {
            x[i] = activate(x[i],actType, param);
        }
    }
    else
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<numX;++i)
        {
            x[i] = activate(x[i],actType, param);
        }
    }
#endif

#ifdef USE_ARM
#ifdef USE_NEON
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<numX/4;++i)
        {
            ActivationsNeon::activateNeon4(x+i*4,actType,param);
        }

        for(int i=(numX/4)*4 ; i<numX;++i)
        {
            x[i] = activate(x[i],actType, param);
        }
#else
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<numX;++i)
        {
            x[i] = activate(x[i],actType, param);
        }
#endif
#endif
}

}
