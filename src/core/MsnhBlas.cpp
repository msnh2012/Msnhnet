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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN; ++i)
        {
            y[i*stepY]  = y[i*stepY] + alpha * x[i*stepX];
        }
    }
#endif
}

void Blas::cpuScale(const int &inputN, const float &alpha, float *const &x, const int &stepX)
{
#ifdef USE_OPEN_BLAS
    cblas_sscal(inputN, alpha, x, stepX);
#else
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
                   const int &batch, const int &filters, const int &outSize)
{

    for(int b=0; b<batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int f=0; f<filters; ++f)
        {
            for(int i=0; i<outSize; ++i)
            {
                int index = b*filters*outSize + f*outSize + i;

                x[index]  = (x[index] - mean[f])/(sqrt(variance[f] + 0.00001f));
            }
        }
    }
}

void Blas::cpuSmoothL1(const int &n, float * const &pred, float * const &truth, float * const &delta, float * const &error)
{
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < n; ++i)
    {
        float diff  = truth[i] - pred[i];
        error[i]    = diff * diff;
        delta[i]    = diff;
    }
}

void Blas::cpuFlatten(float * const &x, const int &size, const int &layers, const int &batch, const int &forward)
{
    float *swapVal  =   new float[static_cast<size_t>(size*layers*batch)]();

    for (int b = 0; b < batch; ++b)
    {
        for (int c = 0; c < layers; ++c)
        {
            for (int i = 0; i < size; ++i)
            {
                int i1  =   b*layers*size + c*size + i;
                int i2  =   b*layers*size + i*layers + c;

                if(forward!=0)
                {
                    swapVal[i2] = x[i1];
                }
                else
                {
                    swapVal[i1] = x[i2];
                }
            }
        }
    }

    memcpy(x, swapVal, static_cast<size_t>(size*layers*batch)*sizeof(float));

    delete[] swapVal;
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
                    float sumExp    = _mm256_cvtss_f32(_mm256_hadd_ps(exp, exp));

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
                    float sumExp    = _mm256_cvtss_f32(_mm256_hadd_ps(exp, exp));
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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
#pragma omp parallel for num_threads(OMP_THREAD)
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

        error[i]    =   (t >0 ) ? -log(p) : 0;
        delta[i]    =   t - p;
    }
}

void Blas::cpuLogisticCorssEntropy(const int &num, float * const &pred, float * const &truth, float * const &delta, float * const &error)
{
    for (int i = 0; i < num; ++i)
    {
        float t     =   truth[i];
        float p     =   pred[i];

        error[i]    =   -t*log(p) - (1-t)*log(1-p);
        delta[i]    =   t - p;
    }
}

void Blas::cpuUpSample(float * const &in, const int &width, const int &height, const int &channel, const int &batch, const int &stride,
                       const int &forward, const float &scale, float * const &out)
{
    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int k = 0; k < channel; ++k)
        {
            for (int j = 0; j < height * stride; ++j)
            {
                for (int i = 0; i < width * stride; ++i)
                {
                    int inIndex     =   b*width*height*channel + k*width*height + (j/stride)*width + i/stride;
                    int outIndex    =   b*width*height*channel*stride*stride + k*width*height*stride*stride + j*width*stride + i;

                    if(forward)
                    {
                        out[outIndex]   =   scale*in[inIndex];
                    }
                    else
                    {
                        in[inIndex]     +=  scale*out[outIndex];
                    }
                }
            }
        }
    }
}

}
