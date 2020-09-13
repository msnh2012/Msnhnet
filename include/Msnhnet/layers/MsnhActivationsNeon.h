#ifndef MSNHACTIONVATIONSNEON_H
#define MSNHACTIONVATIONSNEON_H
#ifdef USE_NEON
#include "Msnhnet/core/MsnhNeonMathEx.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
class MsnhNet_API ActivationsNeon
{
public:
    static inline void logisticActivateSize4(float *const &x)
    {
        const float32x4_t one   =   vdupq_n_f32(1.f);  

        const float32x4_t zero  =   vdupq_n_f32(0);  

        float32x4_t loadX       =   vld1q_f32(x);      

        float32x4_t divDown     =   vaddq_f32(one,exp_ps(vsubq_f32(zero,loadX)));

        float32x4_t rec0        =   vrecpeq_f32(divDown);
        float32x4_t rec1        =   vmulq_f32(rec0,vrecpsq_f32(rec0,divDown));

        float32x4_t result      =   vmulq_f32(one, rec1);

        vst1q_f32(x,result);

    }

    static inline void loggyActivateSize4(float *const &x)
    {
        const float32x4_t one   =   vdupq_n_f32(1.f);  

        const float32x4_t two   =   vdupq_n_f32(2.f);  

        const float32x4_t zero  =   vdupq_n_f32(0);  

        float32x4_t loadX       =   vld1q_f32(x);      

        float32x4_t divDown     =   vaddq_f32(one,exp_ps(vsubq_f32(zero,loadX)));
        float32x4_t rec0        =   vrecpeq_f32(divDown);
        float32x4_t rec1        =   vmulq_f32(rec0,vrecpsq_f32(rec0,divDown));
        float32x4_t first       =   vmulq_f32(two, rec1);

        float32x4_t result      =   vsubq_f32(first, one);
        vst1q_f32(x,result);

    }

    static inline void reluActivateSize4(float *const &x)
    {
        for (int i = 0; i < 4; ++i)
        {
            if(x[i]<0) x[i]=0;
        }
    }

    static inline void relu6ActivateSize4(float *const &x)
    {
        x[0]=(x[0]>0?x[0]:0)>6?6:(x[0]>0?x[0]:0);
        x[1]=(x[1]>0?x[1]:0)>6?6:(x[1]>0?x[1]:0);
        x[2]=(x[2]>0?x[2]:0)>6?6:(x[2]>0?x[2]:0);
        x[3]=(x[3]>0?x[3]:0)>6?6:(x[3]>0?x[3]:0);
    }

    static inline void hardSwishActivateSize4(float *const &x)
    {
        float32x4_t load    = vld1q_f32(x);
        const float32x4_t threeF   =   vdupq_n_f32(3.f);
        const float32x4_t oneDSixF =   vdupq_n_f32(0.16666667f);
        float32x4_t result  = vaddq_f32(load,threeF);
        vst1q_f32(x,result);
        x[0]=(x[0]>0?x[0]:0)>6?6:(x[0]>0?x[0]:0);
        x[1]=(x[1]>0?x[1]:0)>6?6:(x[1]>0?x[1]:0);
        x[2]=(x[2]>0?x[2]:0)>6?6:(x[2]>0?x[2]:0);
        x[3]=(x[3]>0?x[3]:0)>6?6:(x[3]>0?x[3]:0);
        float32x4_t load1   = vld1q_f32(x);
        float32x4_t res     = vmulq_f32(load,load1);
        res                 = vmulq_f32(res,oneDSixF);
        vst1q_f32(x,res);
    }

    static inline void eluActivateSize4(float *const &x)
    {
        const float32x4_t one   =   vdupq_n_f32(1.f);  

        const float32x4_t zero  =   vdupq_n_f32(0);  

        float32x4_t loadX       =   vld1q_f32(x);      

        uint32x4_t a            =   vcgeq_f32(loadX, zero);
        uint32x4_t b            =   vcltq_f32(loadX, zero);
        a                       =   vandq_u32(a, vdupq_n_u32(1));
        b                       =   vandq_u32(b, vdupq_n_u32(1));

        float32x4_t a1          =   vcvtq_f32_u32(a);
        float32x4_t b1          =   vcvtq_f32_u32(b);

        float32x4_t result      =   vaddq_f32(vmulq_f32(a1, loadX), vmulq_f32(b1, vsubq_f32(exp_ps(loadX), one)));
        vst1q_f32(x,result);

    }

    static inline void seluActivateSize4(float *const &x)
    {
        const float32x4_t one   =   vdupq_n_f32(1.f);  

        const float32x4_t zero  =   vdupq_n_f32(0);  

        float32x4_t loadX       =   vld1q_f32(x);      

        uint32x4_t a            =   vcgeq_f32(loadX, zero);
        uint32x4_t b            =   vcltq_f32(loadX, zero);
        a                       =   vandq_u32(a, vdupq_n_u32(1));
        b                       =   vandq_u32(b, vdupq_n_u32(1));

        float32x4_t a1          =   vcvtq_f32_u32(a);
        float32x4_t b1          =   vcvtq_f32_u32(b);

        float32x4_t alpha       =   vdupq_n_f32(1.0507f);
        float32x4_t beta        =   vdupq_n_f32(1.6732f);

        float32x4_t first       =   vmulq_f32(vmulq_f32(a1,alpha),loadX);
        float32x4_t second      =   vmulq_f32(vmulq_f32(vmulq_f32(b1,alpha),beta),vsubq_f32(exp_ps(loadX),one));
        float32x4_t result      =   vaddq_f32(first,second);
        vst1q_f32(x,result);

    }

    static inline void relieActivateSize4(float *const &x)
    {
        const float32x4_t zero  =   vdupq_n_f32(0);  

        const float32x4_t alpha =   vdupq_n_f32(.01f);  

        float32x4_t loadX       =   vld1q_f32(x);      

        uint32x4_t a            =   vcgtq_f32(loadX, zero);
        uint32x4_t b            =   vcleq_f32(loadX, zero);
        a                       =   vandq_u32(a, vdupq_n_u32(1));
        b                       =   vandq_u32(b, vdupq_n_u32(1));

        float32x4_t a1          =   vcvtq_f32_u32(a);
        float32x4_t b1          =   vcvtq_f32_u32(b);

        float32x4_t result      =   vaddq_f32(vmulq_f32(a1,loadX),vmulq_f32(alpha,vmulq_f32(b1,loadX)));
        vst1q_f32(x,result);

    }

    static inline void rampActivateSize4(float *const &x)
    {
        const float32x4_t zero  =   vdupq_n_f32(0);  

        const float32x4_t alpha =   vdupq_n_f32(.1f);  

        float32x4_t loadX       =   vld1q_f32(x);      

        uint32x4_t a            =   vcgtq_f32(loadX, zero);
        a                       =   vandq_u32(a, vdupq_n_u32(1));
        float32x4_t a1          =   vcvtq_f32_u32(a);
        float32x4_t result      =   vaddq_f32(vmulq_f32(a1,loadX),vmulq_f32(alpha,loadX));
        vst1q_f32(x,result);

    }

    static inline void leakyActivateSize4(float *const &x, const float& param = 0.1f)
    {
        const float32x4_t zero  =   vdupq_n_f32(0);  

        const float32x4_t alpha =   vdupq_n_f32(param);  

        float32x4_t loadX       =   vld1q_f32(x);      

        uint32x4_t a            =   vcgtq_f32(loadX, zero);
        uint32x4_t b            =   vcleq_f32(loadX, zero);
        a                       =   vandq_u32(a, vdupq_n_u32(1));
        b                       =   vandq_u32(b, vdupq_n_u32(1));

        float32x4_t a1          =   vcvtq_f32_u32(a);
        float32x4_t b1          =   vcvtq_f32_u32(b);

        float32x4_t result      =   vaddq_f32(vmulq_f32(a1,loadX),vmulq_f32(alpha,vmulq_f32(b1,loadX)));
        vst1q_f32(x,result);

    }

    static inline void tanhActivateSize4(float *const &x)
    {
        const float32x4_t one   =   vdupq_n_f32(1.f);  

        const float32x4_t two   =   vdupq_n_f32(2.f);  

        float32x4_t loadX       =   vld1q_f32(x);      

        float32x4_t up          =   vsubq_f32(exp_ps(vmulq_f32(two,loadX)),one);
        float32x4_t down        =   vaddq_f32(exp_ps(vmulq_f32(two,loadX)),one);

        float32x4_t rec0        =   vrecpeq_f32(down);                      

        float32x4_t rec1        =   vmulq_f32(rec0,vrecpsq_f32(rec0,down)); 

        float32x4_t result      =   vmulq_f32(up, rec1);                    

        vst1q_f32(x,result);

    }

    static inline void stairActivateSize4(float *const &x)
    {
        /*TODO:*/
        for (int i = 0; i < 4; ++i)
        {
            int n = static_cast<int>(floor(x[i]));
            if (n%2 == 0)
            {
                x[i] = static_cast<float>(floor(x[i]/2.f));
            }
            else
            {
                x[i] = static_cast<float>((x[i] - n) + floor(x[i]/2.f));
            }
        }
    }

    static inline void hardtanActivateSize4(float *const &x)
    {
        for (int i = 0; i < 4; ++i)
        {
            if (x[i] < -1)
            {
                x[i] = -1;
            }
            if (x[i] > 1)
            {
                x[i] = 1;
            }
        }
    }

    static inline void softplusActivateSize4(float *const &x, const float &threshold)
    {

        const float32x4_t one   =   vdupq_n_f32(1.f);  

        const float32x4_t alpha =   vdupq_n_f32(threshold);  

        const float32x4_t nAlpha=   vdupq_n_f32(-1.0f*threshold);
        float32x4_t loadX       =   vld1q_f32(x);      

        uint32x4_t a            =   vcgtq_f32(loadX, alpha);
        uint32x4_t b            =   vcltq_f32(loadX, nAlpha);
        uint32x4_t c            =   vandq_u32(vcleq_f32(loadX, alpha),vcgeq_f32(loadX, nAlpha));
        a                       =   vandq_u32(a, vdupq_n_u32(1));
        b                       =   vandq_u32(b, vdupq_n_u32(1));
        c                       =   vandq_u32(c, vdupq_n_u32(1));

        float32x4_t a1          =   vcvtq_f32_u32(a);
        float32x4_t b1          =   vcvtq_f32_u32(b);
        float32x4_t c1          =   vcvtq_f32_u32(c);

        float32x4_t result      =   vaddq_f32(vaddq_f32(vmulq_f32(a1,loadX),vmulq_f32(b1,exp_ps(loadX))),
                                              vmulq_f32(c1,log_ps(vaddq_f32(exp_ps(loadX),one))));
        vst1q_f32(x,result);

    }

    static inline void plseActivateSize4(float *const &x)
    {

        const float32x4_t four  =   vdupq_n_f32(4.f);  

        const float32x4_t nFour =   vdupq_n_f32(-4.f);  

        const float32x4_t one   =   vdupq_n_f32(1.f);  

        const float32x4_t alpha =   vdupq_n_f32(0.01f);  

        const float32x4_t beta  =   vdupq_n_f32(0.125f);  

        const float32x4_t gama  =   vdupq_n_f32(0.5f);  

        float32x4_t loadX       =   vld1q_f32(x);      

        uint32x4_t a            =   vcgtq_f32(loadX, four);
        uint32x4_t b            =   vcltq_f32(loadX, nFour);
        uint32x4_t c            =   vandq_u32(vcleq_f32(loadX, four),vcgeq_f32(loadX, nFour));
        a                       =   vandq_u32(a, vdupq_n_u32(1));
        b                       =   vandq_u32(b, vdupq_n_u32(1));
        c                       =   vandq_u32(c, vdupq_n_u32(1));

        float32x4_t a1          =   vcvtq_f32_u32(a);
        float32x4_t b1          =   vcvtq_f32_u32(b);
        float32x4_t c1          =   vcvtq_f32_u32(c);

        float32x4_t first       =   vmulq_f32(b1, vmulq_f32(alpha, vaddq_f32(loadX,four)));
        float32x4_t second		=   vmulq_f32(a1, vaddq_f32(vmulq_f32(alpha, vaddq_f32(loadX, four)), one));
        float32x4_t third       =   vmulq_f32(c1, vaddq_f32(vmulq_f32(beta,loadX),gama));

        float32x4_t result      =   vaddq_f32(first,vaddq_f32(second,third));
        vst1q_f32(x,result);

    }

    static inline void lhtanActivateSize4(float *const &x)
    {
        const float32x4_t zero  =   vdupq_n_f32(0);  

        const float32x4_t one   =   vdupq_n_f32(1.f);  

        const float32x4_t alpha =   vdupq_n_f32(0.001f);  

        float32x4_t loadX       =   vld1q_f32(x);      

        uint32x4_t a            =   vcgtq_f32(loadX, one); 

        uint32x4_t b            =   vcltq_f32(loadX, zero);
        uint32x4_t c            =   vandq_u32(vcleq_f32(loadX, one),vcgeq_f32(loadX, zero));
        a                       =   vandq_u32(a, vdupq_n_u32(1));
        b                       =   vandq_u32(b, vdupq_n_u32(1));
        c                       =   vandq_u32(c, vdupq_n_u32(1));

        float32x4_t a1          =   vcvtq_f32_u32(a);
        float32x4_t b1          =   vcvtq_f32_u32(b);
        float32x4_t c1          =   vcvtq_f32_u32(c);

        float32x4_t first		=   vmulq_f32(a1, vaddq_f32(vmulq_f32(alpha, vsubq_f32(loadX, one)), one));
        float32x4_t second      =   vmulq_f32(b1, vmulq_f32(alpha, loadX));
        float32x4_t third       =   vmulq_f32(c1, loadX);

        float32x4_t result      =   vaddq_f32(first,vaddq_f32(second,third));
        vst1q_f32(x,result);

    }

    static inline void mishActivateSize4(float *const &x)
    {
        float32x4_t loadX       =   vld1q_f32(x);      

        float32x4_t result      =   vmulq_f32(loadX, tanh_ps(log_ps(vaddq_f32(exp_ps(loadX), vdupq_n_f32(1.f)))));
        vst1q_f32(x,result);

    }

    static inline void swishActivateSize4(float *const &x)
    {
        const float32x4_t one   =   vdupq_n_f32(1.0f);
        float32x4_t loadX       =   vld1q_f32(x);      

        float32x4_t divDown     =   vaddq_f32(one,exp_ps(vsubq_f32(vdupq_n_f32(0), loadX)));

        float32x4_t rec0        =   vrecpeq_f32(divDown);                         

        float32x4_t rec1        =   vmulq_f32(rec0,vrecpsq_f32(rec0,divDown));    

        float32x4_t logistic    =   vmulq_f32(one, rec1);                         

        float32x4_t result      =   vmulq_f32(loadX,logistic);
        vst1q_f32(x,result);

    }

    static void activateNeon4(float * const &x, const ActivationType &actType, const float &params = 0.1f);
};

}
#endif
#endif 

