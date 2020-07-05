#ifndef MSNHACTIVATIONS_H
#define MSNHACTIVATIONS_H
#include <math.h>
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhSimd.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API Activations
{
public:

   static ActivationType getActivation(const std::string &msg);

   static std::string getActivationStr(const ActivationType &type);

   static float activate(const float &x, const ActivationType &actType, const float &params = 0.1f);

   static void activateArray(float *const &x, const int &numX, const ActivationType &actType, const float &param = 0.1f);
    static void activateArrayNormCh(float *const &x, const int &numX, const int &batch, const int &channels, const int &whStep, float *const &output);
    static void activateArrayNormChSoftMax(float *const &x, const int &numX, const int &batch, const int &channels, const int &whStep, float *const &output, const int &useMaxVal);

private:

   static inline float linearActivate(const float &x)
    {
        return x;
    }

   static inline float logisticActivate(const float &x)
    {
        return static_cast<float>(1.f/(1.f + exp(-x)));
    }

   static inline float loggyActivate(const float &x)
    {
        return static_cast<float>(2.f/(1.f + exp(-x)) - 1);
    }

   static inline float reluActivate(const float &x)
    {
        return x*(x>0);
    }

   static inline float relu6Activate(const float &x)
    {
        return (x>0?x:0)>6?6:(x>0?x:0);
    }

   static inline float eluActivate(const float &x)
    {
        return static_cast<float>((x >= 0)*x + (x < 0)*(exp(x)-1));
    }

   static inline float seluActivate(const float &x)
    {
        return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x) - 1);
    }

   static inline float relieActivate(const float &x)
    {
        return (x>0) ? x : .01f*x;
    }

   static inline float rampActivate(const float &x)
    {
        return x*(x>0) + .1f*x;
    }

   static inline float leakyActivate(const float &x, const float& param = 0.1f)
    {
        return (x>0) ? x : param*x;
    }

   static inline float tanhActivate(const float &x)
    {
        return static_cast<float>((exp(2*x)-1)/(exp(2*x)+1));
    }

   static inline float stairActivate(const float &x)
    {
        int n = static_cast<int>(floor(x));
        if (n%2 == 0)
        {
            return static_cast<float>(floor(x/2.f));
        }
        else
        {
            return static_cast<float>((x - n) + floor(x/2.f));
        }
    }

   static inline float hardtanActivate(const float &x)
    {
        if (x < -1)
        {
            return -1;
        }
        if (x > 1)
        {
            return 1;
        }
        return x;
    }

   static inline float softplusActivate(float x, float threshold)
    {
        if (x > threshold)
        {
            return x;
        }
        else if (x < -threshold)
        {
            return expf(x);
        }
        return logf(expf(x) + 1);
    }

   static inline float plseActivate(const float &x)
    {
        if(x < -4)
        {
            return .01f * (x + 4);
        }
        if(x > 4)
        {
            return .01f * (x - 4) + 1;
        }
        return .125f*x + .5f;
    }

   static inline float lhtanActivate(const float &x)
    {
        if(x < 0.0f)
        {
            return .001f*x;
        }
        if(x > 1.0f)
        {
            return .001f*(x-1) + 1;
        }
        return x;
    }

   static inline float mishActivate(const float &x)
    {
        const float mishThreshHold = 20.f;
        return x*tanhf(softplusActivate(x, mishThreshHold));
    }

   static inline float swishActivate(const float &x)
    {
        return x*logisticActivate(x);
    }
};
}

#endif 

