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
    for(int i=0; i<inputN; i++)
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
        for(i=0; i<inputN/4; i++)
        {
            float32x4_t a, b, c, result;
            a = vdupq_n_f32(alpha);
            b = vld1q_f32(x+(i<<2));
            c = vld1q_f32(y+(i<<2));
            result = vmulq_f32(a,b);
            result = vaddq_f32(result, c);
            vst1q_f32(y+(i<<2),result);
        }

       for(int j=(i<<2);j<inputN;j++)
        {
            y[i]  = y[i] + alpha * x[i];
        }
#else
        for(int i=0; i<inputN; i++)
        {
            y[i]  = y[i] + alpha * x[i];
        }
#endif

   }
    else
    {
        for(int i=0; i<inputN; i++)
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
    for(int i=0; i<inputN; i++)
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

void Blas::softmax(float * const &input, const int &num, const float &temprature,  const int &stride, float * const &output)
{
    float sum       =   0;
    float largest   =   -FLT_MAX;

   for (int i = 0; i < num; ++i)
    {
        if(input[i*stride] > largest)
        {
            largest = input[i*stride];
        }
    }

   for (int i = 0; i < num; ++i)
    {
        float e         =   exp(input[i*stride]/temprature - largest/temprature);
        sum             +=  e;
        input[i*stride] =   e;
    }

   for (int i = 0; i < num; ++i)
    {
        output[i*stride] /= sum;
    }
}

void Blas::cpuSoftmax(float * const &input, const int &num, const int &batch, const int &batchOff, const int &groups, const int &groupOff, const float &temperature,  const int &stride,float * const &output)
{
    for (int b = 0; b < batch; ++b)
    {
        for (int g = 0; g < groups; ++g)
        {
            softmax(input + b*batchOff + g*groupOff, num, temperature, stride,  output+b*batchOff+g*groupOff);
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
