#ifndef MSNHACTIVATIONSAVX_H
#define MSNHACTIVATIONSAVX_H
#ifdef USE_X86
#include "Msnhnet/core/MsnhAvxMathEx.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{

class MsnhNet_API ActivationsAvx
{
public:
    const static __m256 c1f;
    const static __m256 c0f;
    const static __m256 c2f;
    const static __m256 c0_1f;
    const static __m256 c0_5f;
    const static __m256 c0_01f;
    const static __m256 c0_001f;
    const static __m256 c0_125f;
    const static __m256 c4f;
    const static __m256 cN4f;
    const static __m256 c3f;
    const static __m256 c0_16f;

    static inline void logisticActivateSize8(float *const &x)
    {
        __m256 loadX     =  _mm256_loadu_ps(x);
        __m256 result    =  _mm256_div_ps( c1f,
                                           _mm256_add_ps(c1f,
                                                         exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), loadX))));
        _mm256_storeu_ps(x,result);

    }

    static inline void loggyActivateSize8(float *const &x)
    {
        __m256 loadX     =  _mm256_loadu_ps(x);
        __m256 result    =  _mm256_sub_ps(_mm256_div_ps(c2f,_mm256_add_ps(c1f,exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), loadX)))),
                                          c1f);
        _mm256_storeu_ps(x,result);

    }

    static inline void reluActivateSize8(float *const &x)
    {
        for (int i = 0; i < 8; ++i)
        {
            if(x[i]<0) x[i]=0;
        }
    }

    static inline void relu6ActivateSize8(float *const &x)
    {
        x[0]=(x[0]>0?x[0]:0)>6?6:(x[0]>0?x[0]:0);
        x[1]=(x[1]>0?x[1]:0)>6?6:(x[1]>0?x[1]:0);
        x[2]=(x[2]>0?x[2]:0)>6?6:(x[2]>0?x[2]:0);
        x[3]=(x[3]>0?x[3]:0)>6?6:(x[3]>0?x[3]:0);
        x[4]=(x[4]>0?x[4]:0)>6?6:(x[4]>0?x[4]:0);
        x[5]=(x[5]>0?x[5]:0)>6?6:(x[5]>0?x[5]:0);
        x[6]=(x[6]>0?x[6]:0)>6?6:(x[6]>0?x[6]:0);
        x[7]=(x[7]>0?x[7]:0)>6?6:(x[7]>0?x[7]:0);
    }

    static inline void hardSwishActivateSize8(float *const &x)
    {
        __m256 load     = _mm256_loadu_ps(x);
        __m256 result   = _mm256_add_ps(load,c3f);
        _mm256_storeu_ps(x,result);
        x[0]=(x[0]>0?x[0]:0)>6?6:(x[0]>0?x[0]:0);
        x[1]=(x[1]>0?x[1]:0)>6?6:(x[1]>0?x[1]:0);
        x[2]=(x[2]>0?x[2]:0)>6?6:(x[2]>0?x[2]:0);
        x[3]=(x[3]>0?x[3]:0)>6?6:(x[3]>0?x[3]:0);
        x[4]=(x[4]>0?x[4]:0)>6?6:(x[4]>0?x[4]:0);
        x[5]=(x[5]>0?x[5]:0)>6?6:(x[5]>0?x[5]:0);
        x[6]=(x[6]>0?x[6]:0)>6?6:(x[6]>0?x[6]:0);
        x[7]=(x[7]>0?x[7]:0)>6?6:(x[7]>0?x[7]:0);
        __m256 load1    = _mm256_loadu_ps(x);
        __m256 res      = _mm256_mul_ps(load,load1);
        res             = _mm256_mul_ps(res,c0_16f);
        _mm256_storeu_ps(x,res);
    }

    static inline void eluActivateSize8(float *const &x)
    {
        __m256 loadX      =  _mm256_loadu_ps(x);
        __m256 a	      = _mm256_cmp_ps(loadX, c0f, _CMP_GE_OQ);
        __m256 b		  = _mm256_cmp_ps(loadX, c0f, _CMP_NGT_UQ);
        a				  = _mm256_and_ps(a, c1f);
        b				  = _mm256_and_ps(b, c1f);

        __m256 result     = _mm256_add_ps(_mm256_mul_ps(a, loadX), _mm256_mul_ps(b, _mm256_sub_ps(exp256_ps(loadX), c1f)));
        _mm256_storeu_ps(x,result);

    }

    static inline void seluActivateSize8(float *const &x)
    {
        __m256 loadX        =   _mm256_loadu_ps(x);
        __m256 a            =   _mm256_cmp_ps(loadX, c0f, _CMP_GE_OQ);
        __m256 b            =   _mm256_cmp_ps(loadX, c0f, _CMP_NGT_UQ);
        a                   =   _mm256_and_ps(a, c1f);
        b                   =   _mm256_and_ps(b, c1f);
        const __m256 alpha  =   _mm256_set1_ps(1.0507f);
        const __m256 beta   =   _mm256_set1_ps(1.6732f);

        __m256       first  =   _mm256_mul_ps(_mm256_mul_ps(a,alpha),loadX);
        __m256       second =   _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(b,alpha),beta),_mm256_sub_ps(exp256_ps(loadX),c1f));
        __m256       result =   _mm256_add_ps(first,second);
        _mm256_storeu_ps(x,result);

    }

    static inline void relieActivateSize8(float *const &x)
    {
        __m256 loadX        =   _mm256_loadu_ps(x);
        __m256 a            =   _mm256_cmp_ps(loadX, c0f, _CMP_GT_OQ);
        __m256 b            =   _mm256_cmp_ps(loadX, c0f, _CMP_NGE_UQ);
        a                   =   _mm256_and_ps(a, c1f);
        b                   =   _mm256_and_ps(b, c1f);
        __m256       result =   _mm256_add_ps(_mm256_mul_ps(a,loadX),_mm256_mul_ps(c0_01f,_mm256_mul_ps(b,loadX)));
        _mm256_storeu_ps(x,result);

    }

    static inline void rampActivateSize8(float *const &x)
    {
        __m256 loadX        =   _mm256_loadu_ps(x);
        __m256 a            =   _mm256_cmp_ps(loadX, c0f, _CMP_GT_OQ);
        a                   =   _mm256_and_ps(a, c1f);
        __m256       result =   _mm256_add_ps(_mm256_mul_ps(a,loadX),_mm256_mul_ps(c0_1f,loadX));
        _mm256_storeu_ps(x,result);

    }

    static inline void leakyActivateSize8(float *const &x, const float& param = 0.1f)
    {
        __m256 loadX        =   _mm256_loadu_ps(x);
        const __m256 alpha  =   _mm256_broadcast_ss(&param);
        __m256 a            =   _mm256_cmp_ps(loadX, c0f, _CMP_GT_OQ);
        __m256 b            =   _mm256_cmp_ps(loadX, c0f, _CMP_NGE_UQ);
        a                   =   _mm256_and_ps(a, c1f);
        b                   =   _mm256_and_ps(b, c1f);
        __m256       result =   _mm256_add_ps(_mm256_mul_ps(a,loadX),_mm256_mul_ps(alpha,_mm256_mul_ps(b,loadX)));
        _mm256_storeu_ps(x,result);

    }

    static inline void tanhActivateSize8(float *const &x)
    {
        __m256 loadX        =   _mm256_loadu_ps(x);
        __m256 up           =   _mm256_sub_ps(exp256_ps(_mm256_mul_ps(c2f,loadX)),c1f);
        __m256 down         =   _mm256_add_ps(exp256_ps(_mm256_mul_ps(c2f,loadX)),c1f);
        __m256 result       =   _mm256_div_ps(up,down);
        _mm256_storeu_ps(x,result);

    }

    static inline void stairActivateSize8(float *const &x)
    {
        /*TODO:*/
        for (int i = 0; i < 8; ++i)
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

    static inline void hardtanActivateSize8(float *const &x)
    {
        for (int i = 0; i < 8; ++i)
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

    static inline void softplusActivateSize8(float *const &x, const float &threshold)
    {
        __m256 loadX        =   _mm256_loadu_ps(x);
        const __m256 alpha  =   _mm256_broadcast_ss(&threshold);
        const float th      =   -1.0f*threshold;
        const __m256 nAlpha =   _mm256_broadcast_ss(&th);
        __m256 a            =   _mm256_cmp_ps(loadX, alpha, _CMP_GT_OQ);
        __m256 b            =   _mm256_cmp_ps(loadX, nAlpha, _CMP_NGT_UQ);
        __m256 c            =   _mm256_and_ps(_mm256_cmp_ps(loadX, alpha, _CMP_NGT_UQ),_mm256_cmp_ps(loadX, nAlpha, _CMP_GT_OQ));
        a                   =   _mm256_and_ps(a, c1f);
        b                   =   _mm256_and_ps(b, c1f);
        c                   =   _mm256_and_ps(c, c1f);

        __m256 result       =   _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(a,loadX),_mm256_mul_ps(b,exp256_ps(loadX))),
                                              _mm256_mul_ps(c,log256_ps(_mm256_add_ps(exp256_ps(loadX),c1f))));
        _mm256_storeu_ps(x,result);

    }

    static inline void plseActivateSize8(float *const &x)
    {
        __m256 loadX        =   _mm256_loadu_ps(x);
        __m256 a            =   _mm256_cmp_ps(loadX, c4f, _CMP_GT_OQ);
        __m256 b            =   _mm256_cmp_ps(loadX, cN4f, _CMP_NGT_UQ);
        __m256 c            =   _mm256_and_ps(_mm256_cmp_ps(loadX, c4f, _CMP_NGT_UQ),_mm256_cmp_ps(loadX, cN4f, _CMP_GT_OQ));
        a                   =   _mm256_and_ps(a, c1f);
        b                   =   _mm256_and_ps(b, c1f);
        c                   =   _mm256_and_ps(c, c1f);

        __m256 first        =   _mm256_mul_ps(b,_mm256_mul_ps(c0_01f,_mm256_add_ps(loadX,c4f)));
        __m256 second		=   _mm256_mul_ps(a, _mm256_add_ps(_mm256_mul_ps(c0_01f, _mm256_add_ps(loadX, c4f)), c1f));
        __m256 third        =   _mm256_mul_ps(c,_mm256_add_ps(_mm256_mul_ps(c0_125f,loadX),c0_5f));

        __m256 result       =   _mm256_add_ps(first,_mm256_add_ps(second,third));
        _mm256_storeu_ps(x,result);

    }

    static inline void lhtanActivateSize8(float *const &x)
    {

        __m256 loadX        =   _mm256_loadu_ps(x);
        __m256 a            =   _mm256_cmp_ps(loadX, c1f, _CMP_GT_OQ);
        __m256 b            =   _mm256_cmp_ps(loadX, c0f, _CMP_NGT_UQ);
        __m256 c            =   _mm256_and_ps(_mm256_cmp_ps(loadX, c1f, _CMP_NGT_UQ),_mm256_cmp_ps(loadX, c0f, _CMP_GT_OQ));
        a                   =   _mm256_and_ps(a, c1f);
        b                   =   _mm256_and_ps(b, c1f);
        c                   =   _mm256_and_ps(c, c1f);

        __m256 first		=   _mm256_mul_ps(a, _mm256_add_ps(_mm256_mul_ps(c0_001f, _mm256_sub_ps(loadX, c1f)), c1f));
        __m256 second       =   _mm256_mul_ps(b, _mm256_mul_ps(c0_001f, loadX));
        __m256 third        =   _mm256_mul_ps(c, loadX);

        __m256 result       =   _mm256_add_ps(first,_mm256_add_ps(second,third));
        _mm256_storeu_ps(x,result);

    }
    static inline __m256 sigmoid_avx(__m256 inputs)
    {
        return _mm256_div_ps(
                   c1f, _mm256_add_ps(c1f, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), inputs))));
    }

    static inline __m256 tanh_avx(__m256 inputs)
    {
        return _mm256_fmsub_ps(sigmoid_avx(_mm256_mul_ps(inputs, c2f)), c2f, c1f);
    }

    static inline void mishActivateSize8(float *const &x)
    {
        __m256 loadX            =   _mm256_loadu_ps(x);
        __m256 result           =   _mm256_mul_ps(loadX, tanh_avx(log256_ps(_mm256_add_ps(exp256_ps(loadX), c1f))));
        _mm256_storeu_ps(x,result);

    }

    static inline void swishActivateSize8(float *const &x)
    {
        __m256 loadX     =  _mm256_loadu_ps(x);
        __m256 result    =  _mm256_mul_ps(loadX,_mm256_div_ps( c1f,_mm256_add_ps(c1f,exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), loadX)))));
        _mm256_storeu_ps(x,result);

    }

    static void activateAvx8(float * const &x, const ActivationType &actType, const float &params = 0.1f);
};
}
#endif
#endif 

