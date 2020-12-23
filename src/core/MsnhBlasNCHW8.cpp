#ifdef USE_X86

#include "Msnhnet/core/MsnhBlasNCHW8.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/core/MsnhAvxMathEx.h"

namespace Msnhnet
{

void BlasNCHW8::cpuNCHWToNCHW8(float * const &org, const int width, const int height, const int channel, const int batch, float * const &dstNCHW8)
{
    int outWidth   = 0;
    int outHeight  = height;
    int outChannel = 0;
    getNCHW8Params(width, height, channel,outWidth, outChannel);
    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int oc = 0; oc < outChannel; ++oc)
        {
            for (int oh = 0; oh < outHeight; ++oh)
            {
                for (int ow = 0; ow < outWidth; ++ow)
                {

                    if(ow%NCHW8_PACK > (channel%NCHW8_PACK-1) && channel%NCHW8_PACK != 0 && (oc+1)*NCHW8_PACK > channel)
                    {
                        dstNCHW8[b*outChannel*outHeight*outWidth + oc*outHeight*outWidth + oh*outWidth + ow] = 0;
                    }
                    else
                    {
                        dstNCHW8[b*outChannel*outHeight*outWidth + oc*outHeight*outWidth + oh*outWidth + ow] = org[b*channel*height*width + oc*NCHW8_PACK*width*height + (ow%NCHW8_PACK)*width*height + ow/NCHW8_PACK +  oh*width];
                    }
                }
            }
        }
    }
}

void BlasNCHW8::cpuNCHW8ToNCHW(float * const &orgNCHW8, const int width, const int height, const int channel, const int outChannel, const int batch, float * const &dst)
{
    if(channel !=  (outChannel%NCHW8_PACK==0?outChannel/NCHW8_PACK:outChannel/NCHW8_PACK+1))
    {
        throw Exception(1,"[NCHW8] nchw and nchw8 channel error!",__FILE__,__LINE__,__FUNCTION__);
    }

    if(width%NCHW8_PACK!=0)
    {
        throw Exception(1,"[NCHW8] nchw8 width error!",__FILE__,__LINE__,__FUNCTION__);
    }

    int outWidth  = width/NCHW8_PACK;
    int outHeight = height;

    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int oc = 0; oc < outChannel; ++oc)
        {
            for (int oh = 0; oh < outHeight; ++oh)
            {
                for (int ow = 0; ow < outWidth; ++ow)
                {
                    dst[b*outChannel*outHeight*outWidth + oc*outHeight*outWidth + oh*outWidth + ow] = orgNCHW8[b*channel*height*width + (oc/NCHW8_PACK)*width*height + (oh*width) + ow*NCHW8_PACK];
                }
            }
        }
    }
}

size_t BlasNCHW8::getNCHW8Params(const int width, const int height, const int channel, int &outWidth, int &outChannel)
{
    outChannel = ((channel%NCHW8_PACK) == 0)?channel/NCHW8_PACK:(channel/NCHW8_PACK+1);
    outWidth   = width*NCHW8_PACK;
    return outWidth*height*outChannel;
}

void BlasNCHW8::cpuFillNCHW8(const int &inputN, const float &alpha, float * const &xNCHW8)
{
    std::cout<<"not test"<<std::endl;
    Blas::cpuFill(inputN,alpha,xNCHW8,1);
}

void BlasNCHW8::cpuAxpyNCHW8(const int &inputN, const float &alpha, float * const &xNCHW8, float * const &yNCHW8)
{
    std::cout<<"not test"<<std::endl;

    if(inputN%NCHW8_PACK!=0)
    {
        throw Exception(1,"[NCHW8] inputN error!",__FILE__,__LINE__,__FUNCTION__);
    }

    __m256 mmAlpha = _mm256_set1_ps(alpha);

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < inputN/NCHW8_PACK; ++i)
    {
        __m256 mmX     = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
        __m256 mmY     = _mm256_loadu_ps(yNCHW8 + i*NCHW8_PACK);

        mmY = _mm256_fmadd_ps(mmX,mmAlpha,mmY);

        _mm256_storeu_ps(yNCHW8 + i*NCHW8_PACK, mmY);
    }
}

void BlasNCHW8::cpuArithmeticNCHW8(const Arithmetic &type, const int &inputN, float * const &xNCHW8, float * const &yNCHW8, float *outNCHW8)
{
    std::cout<<"not test"<<std::endl;

    if(inputN%NCHW8_PACK!=0)
    {
        throw Exception(1,"[NCHW8] inputN error!",__FILE__,__LINE__,__FUNCTION__);
    }

    if(type == Arithmetic::ARITH_ADD)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmY = _mm256_loadu_ps(yNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_add_ps(mmX,mmY);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_SUB)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmY = _mm256_loadu_ps(yNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_sub_ps(mmX,mmY);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_SUB_INV)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmY = _mm256_loadu_ps(yNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_sub_ps(mmY,mmX);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_MUL)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmY = _mm256_loadu_ps(yNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_mul_ps(mmX,mmY);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_DIV)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmY = _mm256_loadu_ps(yNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_div_ps(mmX,mmY);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_DIV_INV)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmY = _mm256_loadu_ps(yNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_div_ps(mmY,mmX);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
}

void BlasNCHW8::cpuArithmeticNCHW8(const Arithmetic &type, const int &inputN, float * const &xNCHW8, const float alpha, float *outNCHW8)
{
    std::cout<<"not test"<<std::endl;

    if(inputN%NCHW8_PACK!=0)
    {
        throw Exception(1,"[NCHW8] inputN error!",__FILE__,__LINE__,__FUNCTION__);
    }

    if(type == Arithmetic::ARITH_ADD)
    {
        __m256 mmAlpha = _mm256_set1_ps(alpha);
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_add_ps(mmX,mmAlpha);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_SUB)
    {
        __m256 mmAlpha = _mm256_set1_ps(alpha);
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_sub_ps(mmX,mmAlpha);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_SUB_INV)
    {
        __m256 mmAlpha = _mm256_set1_ps(alpha);
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_sub_ps(mmAlpha,mmX);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_MUL)
    {
        __m256 mmAlpha = _mm256_set1_ps(alpha);
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_mul_ps(mmX,mmAlpha);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_DIV)
    {
        __m256 mmAlpha = _mm256_set1_ps(alpha);
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_div_ps(mmX,mmAlpha);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_DIV_INV)
    {
        __m256 mmAlpha = _mm256_set1_ps(alpha);
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_div_ps(mmAlpha,mmX);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
}

void BlasNCHW8::cpuScientificNCHW8(const Scientific &type, const int &inputN, float * const &xNCHW8, const float alpha, float *outNCHW8)
{
    std::cout<<"not test"<<std::endl;

    if(type == Scientific::SCI_ABS)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN; ++i)
        {
            outNCHW8[i]  = fabs(xNCHW8[i]);
        }
    }
    else if(type == Scientific::SCI_ACOS)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN; ++i)
        {

            outNCHW8[i]  = acosf(xNCHW8[i]);
        }
    }
    else if(type == Scientific::SCI_ASIN)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN; ++i)
        {
            outNCHW8[i]  = asinf(xNCHW8[i]);
        }
    }
    else if(type == Scientific::SCI_ATAN)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN; ++i)
        {
            outNCHW8[i]  = atanf(xNCHW8[i]);
        }
    }
    else if(type == Scientific::SCI_COS)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = cos256_ps(mmX);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Scientific::SCI_COSH)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN; ++i)
        {
            outNCHW8[i]  = coshf(xNCHW8[i]);
        }
    }
    else if(type == Scientific::SCI_SIN)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = sin256_ps(mmX);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Scientific::SCI_SINH)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN; ++i)
        {
            outNCHW8[i]  = sinhf(xNCHW8[i]);
        }
    }
    else if(type == Scientific::SCI_TAN)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for(int i=0; i<inputN; ++i)
            {
                outNCHW8[i]  = tanf(xNCHW8[i]);
            }
    }
    else if(type == Scientific::SCI_TANH)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN; ++i)
        {
            outNCHW8[i]  = tanf(xNCHW8[i]);
        }
    }
    else if(type == Scientific::SCI_EXP)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = exp256_ps(mmX);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
    else if(type == Scientific::SCI_POW)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for(int i=0; i<inputN; ++i)
            {
                outNCHW8[i]  = powf(xNCHW8[i],alpha);
            }
    }
    else if(type == Scientific::SCI_LOG)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = log256_ps(mmX);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }

    }
    else if(type == Scientific::SCI_LOG10)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN; ++i)
        {
            outNCHW8[i]  = log10f(xNCHW8[i]);
        }
    }
    else if(type == Scientific::SCI_SQRT)
    {

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i=0; i<inputN/NCHW8_PACK; ++i)
        {
            __m256 mmX = _mm256_loadu_ps(xNCHW8 + i*NCHW8_PACK);
            __m256 mmO = _mm256_sqrt_ps(mmX);
            _mm256_storeu_ps(outNCHW8 + i*NCHW8_PACK, mmO);
        }
    }
}

void BlasNCHW8::cpuNormNCHW8(float * const &xNCHW8, float * const &meanNCHW8, float * const &varNCHW8, const int &batch, const int &filtersNCHW8, const float &eps, const int &whSize)
{
    std::cout<<"not test"<<std::endl;

    __m256 epsMM = _mm256_set1_ps(eps);

    for(int b=0; b<batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int f=0; f<filtersNCHW8; ++f)
        {
            __m256 meanMM = _mm256_loadu_ps(meanNCHW8 + f);
            __m256 varMM  = _mm256_loadu_ps(varNCHW8  + f);

            for(int i=0; i<whSize; ++i)
            {
                int index = b*filtersNCHW8*whSize + f*whSize + i;
                __m256 xMM = _mm256_loadu_ps(xNCHW8 + index);
                xMM = _mm256_div_ps(_mm256_sub_ps(xMM,meanMM),_mm256_sqrt_ps(_mm256_add_ps(varMM, epsMM)));

            }
        }
    }
}

void BlasNCHW8::cpuUpSampleNCHW8(float * const &inNCHW8, const int &width, const int &height, const int &channelNCHW8, const int &batch,
                                   const int &strideX, const int &strideY, const float &scale, float * const &outNCHW8)
{
    std::cout<<"not test"<<std::endl;

    __m256 scaleMM = _mm256_set1_ps(scale);

    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int k = 0; k < channelNCHW8; ++k)
        {
            for (int j = 0; j < height * strideY; ++j)
            {
                for (int i = 0; i < width * strideX; ++i)
                {
                    int inIndex     =   b*width*height*channelNCHW8 + k*width*height + (j/strideY)*width + i/strideX;
                    int outIndex    =   b*width*height*channelNCHW8*strideX*strideY + k*width*height*strideX*strideY + j*width*strideX + i;

                    __m256 load     =   _mm256_loadu_ps(inNCHW8+inIndex);
                    load            =   _mm256_mul_ps(load, scaleMM);
                    _mm256_storeu_ps(outNCHW8+outIndex, load);

                }
            }
        }
    }
}

void BlasNCHW8::cpuSoftmaxNCHW8(float * const &input, const int &num, const int &batch, const int &batchOff, const int &groups,
                                  const int &groupOff, const float &temperature, const int &stride, float * const &output)
{
    std::cout<<"not test"<<std::endl;

    if( groups != 1)
    {
        throw Exception(1, "[NCHW8]: softmax multi groups not supported!", __FILE__, __LINE__, __FUNCTION__);
    }
    else
    {
        for (int b = 0; b < batch; ++b)
        {
            for (int g = 0; g < groups; ++g)
            {
                Blas::softmax(input + b*batchOff + g*groupOff, num, temperature, stride,  output+b*batchOff+g*groupOff,true);
            }
        }
    }
}

void BlasNCHW8::cpuBilinearResizeNCHW8(float * const &inNCHW8, const int &width, const int &height, const int &channelNCHW8,
                                          const int &batch, const int &outWidth, const int &outHeight, const int &alignCorners, float * const &outNCHW8)
{

    std::cout<<"not test"<<std::endl;

    if(height<1 || outHeight<1 || width <1 || outWidth <1)
    {
        throw Exception(1,"w*x and outw*outx must > 1",__FILE__, __LINE__, __FUNCTION__);
    }

    const float rHeight = (alignCorners==0)?(1.0f*height/outHeight):(1.0f*(height-1)/(outHeight-1));
    const float rWidth  =  (alignCorners==0)?(1.0f*width/outWidth):(1.0f*(width-1)/(outWidth-1));

    const size_t inSize  = width*height;
    const size_t outSize = outWidth*outHeight;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
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

        __m256 h1LamdMM    = _mm256_set1_ps(h1Lamd);
        __m256 h0LamdMM    = _mm256_set1_ps(h0Lamd);

        const float w1Lamd =  w1r - w1;
        const float w0Lamd =  1.0f - w1Lamd;

        __m256 w1LamdMM    = _mm256_set1_ps(w1Lamd);
        __m256 w0LamdMM    = _mm256_set1_ps(w0Lamd);

        const float *inPtr = inNCHW8  + (h1*width + w1)*NCHW8_PACK;
        float *outPtr      = outNCHW8 + NCHW8_PACK*i;

        for (int c = 0; c < channelNCHW8*batch; ++c)
        {
            const float* inTmp = inPtr + c*inSize*NCHW8_PACK;
            __m256 inTmpMM  = _mm256_loadu_ps(inTmp);
            __m256 inTmpMM1 = _mm256_loadu_ps(inTmp + w1p*NCHW8_PACK);

            __m256 inTmpMM2 = _mm256_loadu_ps(inTmp + h1p*width*NCHW8_PACK);
            __m256 inTmpMM3 = _mm256_loadu_ps(inTmp + (h1p*width + w1p)*NCHW8_PACK);

            __m256 part1MM = _mm256_mul_ps(h0LamdMM, _mm256_add_ps(_mm256_mul_ps(w0LamdMM, inTmpMM) , _mm256_mul_ps(w1LamdMM, inTmpMM1)));
            __m256 part2MM = _mm256_mul_ps(h1LamdMM, _mm256_add_ps(_mm256_mul_ps(w0LamdMM, inTmpMM2), _mm256_mul_ps(w1LamdMM, inTmpMM3)));

            __m256 finalMM = _mm256_add_ps(part1MM,part2MM);

            _mm256_storeu_ps(outPtr + c*outSize*NCHW8_PACK, finalMM);
        }

    }
}

}

#endif
