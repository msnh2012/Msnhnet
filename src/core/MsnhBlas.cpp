#include "Msnhnet/core/MsnhBlas.h"

namespace Msnhnet
{

void Blas::cpuCopy(const int &inputN, float *const &input, const int &inputStep,
                   float *const &output, const int &outputStep)
{
#ifdef USE_OPEN_BLAS
    cblas_scopy(inputN, input, inputStep, output, outputStep);
#else
#ifdef USE_OMP
    uint64_t dataLen   = inputN;
    uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
    for(int i=0; i < inputN ; ++i)
    {
        output[i*outputStep] = input[i*inputStep];
    }
#endif

}

void Blas::cpuFill(const int &inputN, const float &alpha, float *const &x, const int &step)
{

#ifdef USE_OMP
    uint64_t dataLen   = inputN;
    uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
    for(int i=0; i<inputN; ++i)
    {
        x[i*step] = alpha;
    }
}

void Blas::cpuAxpy(const int &inputN, const float &alpha, float *const &x,
                   const int &stepX, float *const &y, const int &stepY)
{

#ifdef USE_OPEN_BLAS
    cblas_saxpy(inputN, alpha, x, stepX, y ,stepY);
#else

    if(stepX == stepY && stepX == 1)
    {
#ifdef USE_NEON
        int i=0;
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(i=0; i<inputN/4; ++i)
        {
            float32x4_t a, b, c, result;
            a = vdupq_n_f32(alpha);
            b = vld1q_f32(x+(i*4));
            c = vld1q_f32(y+(i*4));
            result = vmulq_f32(a,b);
            result = vaddq_f32(result, c);
            vst1q_f32(y+(i*4),result);
        }

        for(int j=(inputN/4)*4;j<inputN;++j)
        {
            y[i]  = y[i] + alpha * x[i];
        }
#else

#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            y[i]  = y[i] + alpha * x[i];
        }
#endif

    }
    else
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            y[i*stepY]  = y[i*stepY] + alpha * x[i*stepX];
        }
    }
#endif
}

void Blas::cpuArithmetic(const Arithmetic &type, const int &inputN, float * const &x, const int &stepX, float * const &y, const int &stepY, float *out, const int &stepOut)
{
    if(type == Arithmetic::ARITH_ADD)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = x[i*stepX] + y[i*stepY];
        }
    }
    else if(type == Arithmetic::ARITH_SUB)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = x[i*stepX] - y[i*stepY];
        }
    }
    else if(type == Arithmetic::ARITH_SUB_INV)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = y[i*stepY] - x[i*stepX];
        }
    }
    else if(type == Arithmetic::ARITH_MUL)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = x[i*stepX] * y[i*stepY];
        }
    }
    else if(type == Arithmetic::ARITH_DIV)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]      = x[i*stepX] / y[i*stepY];
        }
    }
    else if(type == Arithmetic::ARITH_DIV_INV)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]      = y[i*stepY] / x[i*stepX];
        }
    }

}

void Blas::cpuArithmetic(const Arithmetic &type, const int &inputN, float * const &x, const int &stepX, const float alpha, float *out, const int &stepOut)
{
    if(type == Arithmetic::ARITH_ADD)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = x[i*stepX] + alpha;
        }
    }
    else if(type == Arithmetic::ARITH_SUB)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = x[i*stepX] - alpha;
        }
    }
    else if(type == Arithmetic::ARITH_SUB_INV)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = alpha - x[i*stepX];
        }
    }
    else if(type == Arithmetic::ARITH_MUL)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = x[i*stepX] * alpha;
        }
    }
    else if(type == Arithmetic::ARITH_DIV)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]      = x[i*stepX] / alpha;
        }
    }
    else if(type == Arithmetic::ARITH_DIV_INV)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]      = alpha / x[i*stepX];
        }
    }

}

void Blas::cpuScientific(const Scientific &type, const int &inputN, float * const &x, const int &stepX, const float alpha, float *out, const int &stepOut, const bool &supportAvx)
{
    if(type == Scientific::SCI_ABS)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = fabs(x[i*stepX]);
        }
    }
    else if(type == Scientific::SCI_ACOS)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = acosf(x[i*stepX]);
        }
    }
    else if(type == Scientific::SCI_ASIN)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = asinf(x[i*stepX]);
        }
    }
    else if(type == Scientific::SCI_ATAN)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = atanf(x[i*stepX]);
        }
    }
    else if(type == Scientific::SCI_COS)
    {
#ifdef USE_X86
        if(supportAvx && stepOut == 1 && stepX == 1)
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/8; ++i)
            {
                __m256 load         = _mm256_loadu_ps(&x[i*8]);
                load                = cos256_ps(load);
                _mm256_storeu_ps(&out[i*8],load);
            }

            for (int i = (inputN/8)*8; i < inputN; ++i)
            {
                out[i] = cosf(x[i]);
            }
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = cosf(x[i*stepX]);
            }
        }
#endif

#ifdef USE_ARM
        if(stepOut == 1 && stepX == 1)
        {
#ifdef USE_NEON

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/4; ++i)
            {
                float32x4_t load    = vld1q_f32(&x[i*4]);
                load                = cos_ps(load);
                vst1q_f32(&out[i*4],load);
            }

            for (int i = (inputN/4)*4; i < inputN; ++i)
            {
                out[i] = cosf(x[i]);
            }

#else

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i]  = cosf(x[i]);
            }
#endif
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = cosf(x[i*stepX]);
            }
        }
#endif
    }
    else if(type == Scientific::SCI_COSH)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = coshf(x[i*stepX]);
        }
    }
    else if(type == Scientific::SCI_SIN)
    {
#ifdef USE_X86
        if(supportAvx && stepOut == 1 && stepX == 1)
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/8; ++i)
            {
                __m256 load         = _mm256_loadu_ps(&x[i*8]);
                load                = sin256_ps(load);
                _mm256_storeu_ps(&out[i*8],load);
            }

            for (int i = (inputN/8)*8; i < inputN; ++i)
            {
                out[i] = sinf(x[i]);
            }
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = sinf(x[i*stepX]);
            }
        }
#endif

#ifdef USE_ARM
        if(stepOut == 1 && stepX == 1)
        {
#ifdef USE_NEON

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/4; ++i)
            {
                float32x4_t load    = vld1q_f32(&x[i*4]);
                load                = sin_ps(load);
                vst1q_f32(&out[i*4],load);
            }

            for (int i = (inputN/4)*4; i < inputN; ++i)
            {
                out[i] = sinf(x[i]);
            }

#else

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i]  = sinf(x[i]);
            }
#endif
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = sinf(x[i*stepX]);
            }
        }
#endif
    }
    else if(type == Scientific::SCI_SINH)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = sinhf(x[i*stepX]);
        }
    }
    else if(type == Scientific::SCI_TAN)
    {
#ifdef USE_X86
        if(supportAvx && stepOut == 1 && stepX == 1)
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/8; ++i)
            {
                __m256 load         = _mm256_loadu_ps(&x[i*8]);
                load                = _mm256_div_ps(sin256_ps(load),cos256_ps(load));
                _mm256_storeu_ps(&out[i*8],load);
            }

            for (int i = (inputN/8)*8; i < inputN; ++i)
            {
                out[i] = tanf(x[i]);
            }
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = tanf(x[i*stepX]);
            }
        }
#endif

#ifdef USE_ARM
        if(stepOut == 1 && stepX == 1)
        {
#ifdef USE_NEON

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/4; ++i)
            {
                float32x4_t load    = vld1q_f32(&x[i*4]);
#ifdef __aarch64__
                load                = vdivq_f32(sin_ps(load),cos_ps(load));
#else
                float32x4_t recip0  = vrecpeq_f32(cos_ps(load));
                float32x4_t recip1  = vmulq_f32(recip0, vrecpsq_f32(recip0,(cos_ps(load))));
                load                = vmulq_f32(sin_ps(load), recip1);
#endif
                vst1q_f32(&out[i*4],load);
            }

            for (int i = (inputN/4)*4; i < inputN; ++i)
            {
                out[i] = tanf(x[i]);
            }

#else

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i]  = tanf(x[i]);
            }
#endif
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = tanf(x[i*stepX]);
            }
        }
#endif
    }
    else if(type == Scientific::SCI_TANH)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = tanf(x[i*stepX]);
        }
    }
    else if(type == Scientific::SCI_EXP)
    {
#ifdef USE_X86
        if(supportAvx && stepOut == 1 && stepX == 1)
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/8; ++i)
            {
                __m256 load         = _mm256_loadu_ps(&x[i*8]);
                load                = exp256_ps(load);
                _mm256_storeu_ps(&out[i*8],load);
            }

            for (int i = (inputN/8)*8; i < inputN; ++i)
            {
                out[i] = expf(x[i]);
            }
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = expf(x[i*stepX]);
            }
        }
#endif

#ifdef USE_ARM
        if(stepOut == 1 && stepX == 1)
        {
#ifdef USE_NEON

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/4; ++i)
            {
                float32x4_t load    = vld1q_f32(&x[i*4]);
                load                = exp_ps(load);
                vst1q_f32(&out[i*4],load);
            }

            for (int i = (inputN/4)*4; i < inputN; ++i)
            {
                out[i] = expf(x[i]);
            }

#else

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i]  = expf(x[i]);
            }
#endif
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = expf(x[i*stepX]);
            }
        }
#endif
    }
    else if(type == Scientific::SCI_POW)
    {
#ifdef USE_X86
        if(supportAvx && stepOut == 1 && stepX == 1)
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/8; ++i)
            {
                __m256 load         = _mm256_loadu_ps(&x[i*8]);
                load                = pow_ps(load, _mm256_set1_ps(alpha));
                _mm256_storeu_ps(&out[i*8],load);
            }

            for (int i = (inputN/8)*8; i < inputN; ++i)
            {
                out[i] = powf(x[i],alpha);
            }
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = powf(x[i*stepX],alpha);
            }
        }
#endif

#ifdef USE_ARM
        if(stepOut == 1 && stepX == 1)
        {
#ifdef USE_NEON

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/4; ++i)
            {
                float32x4_t load    = vld1q_f32(&x[i*4]);
                load                = pow_ps(load,vdupq_n_f32(alpha));
                vst1q_f32(&out[i*4],load);
            }

            for (int i = (inputN/4)*4; i < inputN; ++i)
            {
                out[i] = powf(x[i],alpha);
            }

#else

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i]  = powf(x[i],alpha);
            }
#endif
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = powf(x[i*stepX],alpha);
            }
        }
#endif
    }
    else if(type == Scientific::SCI_LOG)
    {
#ifdef USE_X86
        if(supportAvx && stepOut == 1 && stepX == 1)
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/8; ++i)
            {
                __m256 load         = _mm256_loadu_ps(&x[i*8]);
                load                = log256_ps(load);
                _mm256_storeu_ps(&out[i*8],load);
            }

            for (int i = (inputN/8)*8; i < inputN; ++i)
            {
                out[i] = logf(x[i]);
            }
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = logf(x[i*stepX]);
            }
        }
#endif

#ifdef USE_ARM
        if(stepOut == 1 && stepX == 1)
        {
#ifdef USE_NEON

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/4; ++i)
            {
                float32x4_t load    = vld1q_f32(&x[i*4]);
                load                = log_ps(load);
                vst1q_f32(&out[i*4],load);
            }

            for (int i = (inputN/4)*4; i < inputN; ++i)
            {
                out[i] = logf(x[i]);
            }

#else

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i]  = logf(x[i]);
            }
#endif
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = logf(x[i*stepX]);
            }
        }
#endif
    }
    else if(type == Scientific::SCI_LOG10)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            out[i*stepOut]  = log10f(x[i*stepX]);
        }
    }
    else if(type == Scientific::SCI_SQRT)
    {
#ifdef USE_X86
        if(supportAvx && stepOut == 1 && stepX == 1)
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/8; ++i)
            {
                __m256 load         = _mm256_loadu_ps(&x[i*8]);
                load                = _mm256_sqrt_ps(load);
                _mm256_storeu_ps(&out[i*8],load);
            }

            for (int i = (inputN/8)*8; i < inputN; ++i)
            {
                out[i] = sqrtf(x[i]);
            }
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = sqrtf(x[i*stepX]);
            }
        }
#endif

#ifdef USE_ARM
        if(stepOut == 1 && stepX == 1)
        {
#ifdef USE_NEON

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < inputN/4; ++i)
            {
                float32x4_t load    = vld1q_f32(&x[i*4]);
#ifdef __aarch64__
                load                = vsqrtq_f32(load);
#else
                load                = vrsqrteq_f32(load);
                load                = vrecpeq_f32(load);
#endif
                vst1q_f32(&out[i*4],load);
            }

            for (int i = (inputN/4)*4; i < inputN; ++i)
            {
                out[i] = sqrtf(x[i]);
            }

#else

#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i]  = sqrtf(x[i]);
            }
#endif
        }
        else
        {
#ifdef USE_OMP
            uint64_t dataLen   = inputN;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for(int i=0; i<inputN; ++i)
            {
                out[i*stepOut]  = sqrtf(x[i*stepX]);
            }
        }
#endif
    }

}

void Blas::cpuScale(const int &inputN, const float &alpha, float *const &x, const int &stepX)
{
#ifdef USE_OPEN_BLAS
    cblas_sscal(inputN, alpha, x, stepX);
#else
#ifdef USE_OMP
    uint64_t dataLen   = inputN;
    uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
    for(int i=0; i<inputN; ++i)
    {
        x[i*stepX] = x[i*stepX] * alpha;
    }
#endif
}

void Blas::cpuMean(float *const &x, const int &batch, const int &filters, const int &outSize, float *const &mean)
{
    float scale  =  1.f/(batch * outSize); 

#ifdef USE_OMP
    uint64_t dataLen   = filters*batch*outSize;
    uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
    for(int i=0; i<filters; ++i) 

    {
        mean[i]  = 0;

        for(int j=0; j<batch; ++j) 

        {
            for(int k=0; k<outSize; ++k) 

            {

                int index = j*filters*outSize + i*outSize + k;
                mean[index] += x[index];
            }
        }

        mean[i] = mean[i]*scale;
    }

}

void Blas::cpuVariance(float *const &x, float *const &mean, const int &batch,
                       const int &filters, const int &outSize, float *const &variance)
{
    float scale  =  1.f/(batch * outSize -1);

#ifdef USE_OMP
    uint64_t dataLen   = filters*batch*outSize;
    uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
    for(int i=0; i<filters; ++i)
    {
        variance[i] = 0;

        for(int j=0; j<batch; ++j)
        {
            for(int k=0; k<outSize;++k)
            {
                int index = j*filters*outSize + i*outSize + k;

                variance[i] = variance[i] + pow((x[index] - mean[i]),2);
            }
        }
        variance[i] = variance[i] * scale;
    }

}

void Blas::cpuNorm(float *const &x, float *const &mean, float *const &variance,
                   const int &batch, const int &filters, const float &eps, const int &outSize)
{

    for(int b=0; b<batch; ++b)
    {
#ifdef USE_OMP
        uint64_t dataLen   = filters*outSize;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int f=0; f<filters; ++f)
        {
            for(int i=0; i<outSize; ++i)
            {
                int index = b*filters*outSize + f*outSize + i;

                x[index]  = (x[index] - mean[f])/(sqrt(variance[f] + eps));
            }
        }
    }
}

void Blas::cpuSmoothL1(const int &n, float * const &pred, float * const &truth, float * const &delta, float * const &error)
{
#ifdef USE_OMP
    uint64_t dataLen   = n;
    uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
    for (int i = 0; i < n; ++i)
    {
        float diff      = truth[i] - pred[i];
        float absVal    = fabs(diff);

        if(absVal < 1)
        {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else
        {
            error[i] = 2*absVal - 1.f;
            delta[i] = (diff > 0)? 1.f : -1.f;
        }
    }
}

void Blas::cpuL1(const int &n, float * const &pred, float * const &truth, float * const &delta, float * const &error)
{
#ifdef USE_OMP
    uint64_t dataLen   = n;
    uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
    for (int i = 0; i < n; ++i)
    {
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = diff > 0 ? 1.f : -1.f;
    }
}

void Blas::cpuL2(const int &n, float * const &pred, float * const &truth, float * const &delta, float * const &error)
{
#ifdef USE_OMP
    uint64_t dataLen   = n;
    uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
    for (int i = 0; i < n; ++i)
    {
        float diff  = truth[i] - pred[i];
        error[i]    = diff * diff;
        delta[i]    = diff;
    }
}

void Blas::softmax(float * const &input, const int &num, const float &temperature,  const int &stride, float * const &output, const bool &useAvx)
{
    float sum       =   0;
    float largest   =   -FLT_MAX;

    if(stride == 1)
    {
        float tempLarge = *max_element(input,input+num);
        if(tempLarge > largest)
        {
            largest = tempLarge;
        }

#ifdef USE_X86
        if(useAvx)
        {
            if(temperature == 1)
            {
                for (int i = 0; i < num/8; ++i)
                {
                    __m256 load         = _mm256_loadu_ps(&input[i*8]);
                    __m256 largestMM    = _mm256_broadcast_ss(&largest);
                    __m256 exp          = exp256_ps(_mm256_sub_ps(load,largestMM));
                    _mm256_storeu_ps(&output[i*8],exp);

                    exp = _mm256_add_ps(exp, _mm256_permute2f128_ps(exp, exp, 1));
                    exp = _mm256_hadd_ps(exp, exp);
#ifdef WIN32
                    float sumExp    = _mm256_cvtss_f32(_mm256_hadd_ps(exp, exp));
#else
                    float sumExp    = _mm256_hadd_ps(exp, exp)[0];
#endif
                    sum = sum + sumExp;
                }

                for (int i = (num/8)*8; i < num; ++i)
                {
                    float e         =   exp(input[i] - largest);
                    sum             +=  e;
                    output[i] =   e;
                }

            }
            else
            {
                __m256 tempertureMM = _mm256_broadcast_ss(&temperature);
                __m256 expB         = _mm256_div_ps(_mm256_broadcast_ss(&largest),tempertureMM);

                for (int i = 0; i < num/8; ++i)
                {
                    __m256 load     = _mm256_loadu_ps(&input[i*8]);
                    __m256 expA     = _mm256_div_ps(load,tempertureMM);
                    __m256 exp      = exp256_ps(_mm256_sub_ps(expA,expB));
                    _mm256_storeu_ps(&output[i*8],exp);

                    exp = _mm256_add_ps(exp, _mm256_permute2f128_ps(exp, exp, 1));
                    exp = _mm256_hadd_ps(exp, exp);
#ifdef WIN32
                    float sumExp    = _mm256_cvtss_f32(_mm256_hadd_ps(exp, exp));
#else
                    float sumExp    = _mm256_hadd_ps(exp, exp)[0];
#endif
                    sum = sum + sumExp;
                }

                for (int i = (num/8)*8; i < num; ++i)
                {
                    float e         =   exp(input[i]/temperature - largest/temperature);
                    sum             +=  e;
                    output[i] =   e;
                }
            }

            __m256 sumNN = _mm256_broadcast_ss(&sum);

#ifdef USE_OMP
            uint64_t dataLen   = num;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < num/8; ++i)
            {
                __m256 load     = _mm256_loadu_ps(&output[i*8]);
                load            = _mm256_div_ps(load, sumNN);
                _mm256_storeu_ps(&output[i*8],load);
            }

            for (int i = (num/8)*8; i < num; ++i)
            {
                output[i] /= sum;
            }

        }
        else
        {
            for (int i = 0; i < num; ++i)
            {
                float e         =   exp(input[i]/temperature - largest/temperature);
                sum             +=  e;
                output[i] =   e;
            }

#ifdef USE_OMP
            uint64_t dataLen   = num;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < num; ++i)
            {
                output[i] /= sum;
            }
        }
#endif

#ifdef USE_ARM
        std::cout<<"softmax arm need test"<<std::endl;
        getchar();
#ifdef USE_NEON
        if(temperature == 1)
        {
            for (int i = 0; i < num/4; ++i)
            {
                float32x4_t load        =   vld1q_f32(&input[i*4]);
                float32x4_t largestMM   =   vdupq_n_f32(largest);

                float32x4_t exp         =   exp_ps(vsubq_f32(load,largestMM));

                vst1q_f32(&output[i*4],exp);

#ifdef __arrch64__
                float sumExp            = vaddvq_f32(exp);
#else
                float32x2_t exp2        = vadd_f32(vget_low_f32(exp), vget_high_f32(exp));

                exp2                    = vpadd_f32(exp2, exp2);

                float sumExp            = vget_lane_f32(exp2, 0);

#endif
                sum = sum + sumExp;
            }

            for (int i = (num/4)*4; i < num; ++i)
            {
                float e         =   exp(input[i] - largest);
                sum             +=  e;
                output[i] =   e;
            }
        }
        else
        {
            float32x4_t tempertureMM    =   vdupq_n_f32(temperature);
            float32x4_t largestMM       =   vdupq_n_f32(largest);

            float32x4_t recip0          =   vrecpeq_f32(tempertureMM);
            float32x4_t recip1          =   vmulq_f32(recip0, vrecpsq_f32(recip0, tempertureMM));

            float32x4_t expB            =   vmulq_f32(largestMM, recip1);

            for (int i = 0; i < num/4; ++i)
            {
                float32x4_t load        =   vld1q_f32(&input[i*4]);
                float32x4_t expA        =   vmulq_f32(load,recip1);

                float32x4_t exp         =   exp_ps(vsubq_f32(expA,expB));

                vst1q_f32(&output[i*4],exp);

#ifdef __arrch64__
                float sumExp            = vaddvq_f32(exp);
#else
                float32x2_t exp2        = vadd_f32(vget_low_f32(exp), vget_high_f32(exp));

                exp2                    = vpadd_f32(exp2, exp2);

                float sumExp            = vget_lane_f32(exp2, 0);

#endif
                sum = sum + sumExp;
            }

            for (int i = (num/4)*4; i < num; ++i)
            {
                float e         =   exp(input[i]/temperature - largest/temperature);
                sum             +=  e;
                output[i] =   e;
            }
        }

        float32x4_t sumNN       = vdupq_n_f32(sum);

#ifdef USE_OMP
        uint64_t dataLen   = num;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < num/4; ++i)
        {
            float32x4_t load    = vld1q_f32(&output[i*4]);

            float32x4_t recip00 = vrecpeq_f32(sumNN);
            float32x4_t recip11 = vmulq_f32(recip00, vrecpsq_f32(recip00, sumNN));

            load                = vmulq_f32(load, recip11);
            vst1q_f32(&output[i*4],load);
        }

        for (int i = (num/4)*4; i < num; ++i)
        {
            output[i] /= sum;
        }
#else
        for (int i = 0; i < num; ++i)
        {
            float e         =   exp(input[i]/temperature - largest/temperature);
            sum             +=  e;
            input[i] =   e;
        }

#ifdef USE_OMP
        uint64_t dataLen   = num;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < num; ++i)
        {
            output[i] /= sum;
        }
#endif

#endif

    }
    else
    {
#ifdef USE_OMP
        uint64_t dataLen   = num;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < num; ++i)
        {
            if(input[i*stride] > largest)
            {
                largest = input[i*stride];
            }
        }

        for (int i = 0; i < num; ++i)
        {
            float e         =   exp(input[i*stride]/temperature - largest/temperature);
            sum             +=  e;
            input[i*stride] =   e;
        }

#ifdef USE_OMP
        dataLen   = num;
        threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < num; ++i)
        {
            output[i*stride] /= sum;
        }
    }
}

void Blas::cpuSoftmax(float * const &input, const int &num, const int &batch, const int &batchOff, const int &groups,
                      const int &groupOff, const float &temperature,  const int &stride, float * const &output, const bool &useAvx)
{
    for (int b = 0; b < batch; ++b)
    {
        for (int g = 0; g < groups; ++g)
        {
            softmax(input + b*batchOff + g*groupOff, num, temperature, stride,  output+b*batchOff+g*groupOff, useAvx);
        }
    }
}

void Blas::cpuSoftMaxCrossEntropy(const int &num, float * const &pred, float * const &truth, float * const &delta, float * const &error)
{
    for (int i = 0; i < num; ++i)
    {
        float t     =   truth[i];
        float p     =   pred[i];

        error[i]    =   (t >0 ) ? -logf(p) : 0;
        delta[i]    =   t - p;
    }
}

void Blas::cpuLogisticCorssEntropy(const int &num, float * const &pred, float * const &truth, float * const &delta, float * const &error)
{
    for (int i = 0; i < num; ++i)
    {
        float t     =   truth[i];
        float p     =   pred[i];

        error[i]    =   -t*logf(p) - (1-t)*logf(1-p);
        delta[i]    =   t - p;
    }
}

void Blas::cpuUpSample(float * const &in, const int &width, const int &height, const int &channel, const int &batch, const int &strideX,
                       const int &strideY, const float &scale, float * const &out)
{

    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
        uint64_t dataLen   = channel * height * strideY * width * strideX;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int k = 0; k < channel; ++k)
        {
            for (int j = 0; j < height * strideY; ++j)
            {
                for (int i = 0; i < width * strideX; ++i)
                {
                    int inIndex     =   b*width*height*channel + k*width*height + (j/strideY)*width + i/strideX;
                    int outIndex    =   b*width*height*channel*strideX*strideY + k*width*height*strideX*strideY + j*width*strideX + i;

                    out[outIndex]   =   scale*in[inIndex];
                }
            }
        }
    }
}

void Blas::cpuBilinearResize(float * const &in, const int &width, const int &height, const int &channel, const int &batch, const int &outWidth,
                             const int &outHeight, const int &alignCorners, float * const &out)
{

    if(height<1 || outHeight<1 || width <1 || outWidth <1)
    {
        throw Exception(1,"w*x and outw*outx must > 1",__FILE__, __LINE__, __FUNCTION__);
    }

    const float rHeight = (alignCorners==0)?(1.0f*height/outHeight):(1.0f*(height-1)/(outHeight-1));
    const float rWidth  =  (alignCorners==0)?(1.0f*width/outWidth):(1.0f*(width-1)/(outWidth-1));

    const size_t inSize  = width*height;
    const size_t outSize = outWidth*outHeight;

#ifdef USE_OMP
    uint64_t dataLen   = outWidth;
    uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
    for (int i = 0; i < outSize; ++i)
    {
        const int h2 = i / outWidth;
        const int w2 = i % outWidth;
        float h1r = 0;
        float w1r = 0;

        if(alignCorners==0)
        {
            const float inIdxH =  rHeight*(h2+0.5f)-0.5f;
            h1r = inIdxH<0?0:inIdxH;

            const float inIdxW =  rWidth*(w2+0.5f)-0.5f;
            w1r = inIdxW<0?0:inIdxW;
        }
        else
        {
            h1r = rHeight*h2;
            w1r = rWidth *w2;
        }

        const int h1 = static_cast<int>(h1r);
        const int w1 = static_cast<int>(w1r);

        const int h1p = (h1 < (height - 1))?1:0;
        const int w1p = (w1 < (width  - 1))?1:0;

        const float h1Lamd =  h1r - h1;
        const float h0Lamd =  1.0f - h1Lamd;

        const float w1Lamd =  w1r - w1;
        const float w0Lamd =  1.0f - w1Lamd;

        const float *inPtr = in  + h1*width + w1;
        float *outPtr      = out + i;

        for (int c = 0; c < channel*batch; ++c)
        {
            const float* inTmp = inPtr + c*inSize;
            *(outPtr + c*outSize) = h0Lamd * (w0Lamd*(*inTmp) + w1Lamd*(*(inTmp + w1p)))
                    +h1Lamd * (w0Lamd*(*(inTmp + h1p*width))+w1Lamd * (*(inTmp + h1p*width + w1p)));
        }

    }

}

}
