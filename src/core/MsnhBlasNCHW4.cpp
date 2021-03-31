#include "Msnhnet/core/MsnhBlasNCHW4.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/core/sse/sse_mathfun_extension.h"

namespace Msnhnet
{

void BlasNCHW4::cpuNCHWToNCHW4(float * const &org, const int width, const int height, const int channel, const int batch, float * const &dstNCHW4)
{
    int outWidth   = 0;
    int outHeight  = height;
    int outChannel = 0;
    getNCHW4Params(width, height, channel,outWidth, outChannel);
    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
        uint64_t dataLen   = outChannel*outHeight*outWidth;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int oc = 0; oc < outChannel; ++oc)
        {
            for (int oh = 0; oh < outHeight; ++oh)
            {
                for (int ow = 0; ow < outWidth; ++ow)
                {

                    if(ow%NCHW4_PACK > (channel%NCHW4_PACK-1) && channel%NCHW4_PACK != 0 && (oc+1)*NCHW4_PACK > channel)
                    {
                        dstNCHW4[b*outChannel*outHeight*outWidth + oc*outHeight*outWidth + oh*outWidth + ow] = 0;
                    }
                    else
                    {
                        dstNCHW4[b*outChannel*outHeight*outWidth + oc*outHeight*outWidth + oh*outWidth + ow] = org[b*channel*height*width + oc*NCHW4_PACK*width*height + (ow%NCHW4_PACK)*width*height + ow/NCHW4_PACK +  oh*width];
                    }
                }
            }
        }
    }
}

void BlasNCHW4::cpuNCHW4ToNCHW(float * const &orgNCHW4, const int width, const int height, const int channel, const int outChannel, const int batch, float * const &dst)
{
    if(channel !=  (outChannel%NCHW4_PACK==0?outChannel/NCHW4_PACK:outChannel/NCHW4_PACK+1))
    {
        throw Exception(1,"[NCHW4] nchw and nchw4 channel error!",__FILE__,__LINE__,__FUNCTION__);
    }

    if(width%NCHW4_PACK!=0)
    {
        throw Exception(1,"[NCHW4] nchw4 width error!",__FILE__,__LINE__,__FUNCTION__);
    }

    int outWidth  = width/NCHW4_PACK;
    int outHeight = height;

    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
        uint64_t dataLen   = outChannel*outHeight*outWidth;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int oc = 0; oc < outChannel; ++oc)
        {
            for (int oh = 0; oh < outHeight; ++oh)
            {
                for (int ow = 0; ow < outWidth; ++ow)
                {
                    dst[b*outChannel*outHeight*outWidth + oc*outHeight*outWidth + oh*outWidth + ow] = orgNCHW4[b*channel*height*width + (oc/NCHW4_PACK)*width*height + (oh*width) + ow*NCHW4_PACK];
                }
            }
        }
    }
}

size_t BlasNCHW4::getNCHW4Params(const int width, const int height, const int channel, int &outWidth, int &outChannel)
{
    outChannel = ((channel%NCHW4_PACK) == 0)?channel/NCHW4_PACK:(channel/NCHW4_PACK+1);
    outWidth   = width*NCHW4_PACK;
    return outWidth*height*outChannel;
}

void BlasNCHW4::cpuFillNCHW4(const int &inputN, const float &alpha, float * const &xNCHW4)
{
    std::cout<<"not test"<<std::endl;
    Blas::cpuFill(inputN,alpha,xNCHW4,1);
}

void BlasNCHW4::cpuAxpyNCHW4(const int &inputN, const float &alpha, float * const &xNCHW4, float * const &yNCHW4)
{
    std::cout<<"not test"<<std::endl;

    if(inputN%NCHW4_PACK!=0)
    {
        throw Exception(1,"[NCHW4] inputN error!",__FILE__,__LINE__,__FUNCTION__);
    }

    __m128 mmAlpha = _mm_set1_ps(alpha);

#ifdef USE_OMP
    uint64_t dataLen   = inputN;
    uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
    for (int i = 0; i < inputN/NCHW4_PACK; ++i)
    {
        __m128 mmX     = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
        __m128 mmY     = _mm_loadu_ps(yNCHW4 + i*NCHW4_PACK);

        mmY = _mm_fmadd_ps(mmX,mmAlpha,mmY);

        _mm_storeu_ps(yNCHW4 + i*NCHW4_PACK, mmY);
    }
}

void BlasNCHW4::cpuArithmeticNCHW4(const Arithmetic &type, const int &inputN, float * const &xNCHW4, float * const &yNCHW4, float *outNCHW4)
{
    std::cout<<"not test"<<std::endl;

    if(inputN%NCHW4_PACK!=0)
    {
        throw Exception(1,"[NCHW4] inputN error!",__FILE__,__LINE__,__FUNCTION__);
    }

    if(type == Arithmetic::ARITH_ADD)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmY = _mm_loadu_ps(yNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_add_ps(mmX,mmY);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_SUB)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmY = _mm_loadu_ps(yNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_sub_ps(mmX,mmY);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_SUB_INV)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmY = _mm_loadu_ps(yNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_sub_ps(mmY,mmX);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_MUL)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmY = _mm_loadu_ps(yNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_mul_ps(mmX,mmY);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_DIV)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmY = _mm_loadu_ps(yNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_div_ps(mmX,mmY);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_DIV_INV)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmY = _mm_loadu_ps(yNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_div_ps(mmY,mmX);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
}

void BlasNCHW4::cpuArithmeticNCHW4(const Arithmetic &type, const int &inputN, float * const &xNCHW4, const float alpha, float *outNCHW4)
{
    std::cout<<"not test"<<std::endl;

    if(inputN%NCHW4_PACK!=0)
    {
        throw Exception(1,"[NCHW4] inputN error!",__FILE__,__LINE__,__FUNCTION__);
    }

    if(type == Arithmetic::ARITH_ADD)
    {
        __m128 mmAlpha = _mm_set1_ps(alpha);
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_add_ps(mmX,mmAlpha);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_SUB)
    {
        __m128 mmAlpha = _mm_set1_ps(alpha);
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_sub_ps(mmX,mmAlpha);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_SUB_INV)
    {
        __m128 mmAlpha = _mm_set1_ps(alpha);
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_sub_ps(mmAlpha,mmX);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_MUL)
    {
        __m128 mmAlpha = _mm_set1_ps(alpha);
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_mul_ps(mmX,mmAlpha);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_DIV)
    {
        __m128 mmAlpha = _mm_set1_ps(alpha);
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_div_ps(mmX,mmAlpha);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Arithmetic::ARITH_DIV_INV)
    {
        __m128 mmAlpha = _mm_set1_ps(alpha);
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_div_ps(mmAlpha,mmX);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
}

void BlasNCHW4::cpuScientificNCHW4(const Scientific &type, const int &inputN, float * const &xNCHW4, const float alpha, float *outNCHW4)
{
    std::cout<<"not test"<<std::endl;

    if(type == Scientific::SCI_ABS)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            outNCHW4[i]  = fabs(xNCHW4[i]);
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

            outNCHW4[i]  = acosf(xNCHW4[i]);
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
            outNCHW4[i]  = asinf(xNCHW4[i]);
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
            outNCHW4[i]  = atanf(xNCHW4[i]);
        }
    }
    else if(type == Scientific::SCI_COS)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = cos_ps(mmX);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
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
            outNCHW4[i]  = coshf(xNCHW4[i]);
        }
    }
    else if(type == Scientific::SCI_SIN)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = sin_ps(mmX);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
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
            outNCHW4[i]  = sinhf(xNCHW4[i]);
        }
    }
    else if(type == Scientific::SCI_TAN)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            outNCHW4[i]  = tanf(xNCHW4[i]);
        }
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
            outNCHW4[i]  = tanf(xNCHW4[i]);
        }
    }
    else if(type == Scientific::SCI_EXP)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = exp_ps(mmX);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }
    }
    else if(type == Scientific::SCI_POW)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN; ++i)
        {
            outNCHW4[i]  = powf(xNCHW4[i],alpha);
        }
    }
    else if(type == Scientific::SCI_LOG)
    {
#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = log_ps(mmX);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }

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
            outNCHW4[i]  = log10f(xNCHW4[i]);
        }
    }
    else if(type == Scientific::SCI_SQRT)
    {

#ifdef USE_OMP
        uint64_t dataLen   = inputN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for(int i=0; i<inputN/NCHW4_PACK; ++i)
        {
            __m128 mmX = _mm_loadu_ps(xNCHW4 + i*NCHW4_PACK);
            __m128 mmO = _mm_sqrt_ps(mmX);
            _mm_storeu_ps(outNCHW4 + i*NCHW4_PACK, mmO);
        }

    }
}

void BlasNCHW4::cpuNormNCHW4(float * const &xNCHW4, float * const &meanNCHW4, float * const &varNCHW4,
                             const int &batch, const int &filtersNCHW4, const float &eps, const int &whSize)
{

    std::cout<<"not test"<<std::endl;

    __m128 epsMM = _mm_set1_ps(eps);

    for(int b=0; b<batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int f=0; f<filtersNCHW4; ++f)
        {
            __m128 meanMM = _mm_loadu_ps(meanNCHW4 + f);
            __m128 varMM  = _mm_loadu_ps(varNCHW4  + f);

            for(int i=0; i<whSize; ++i)
            {
                int index = b*filtersNCHW4*whSize + f*whSize + i;
                __m128 xMM = _mm_loadu_ps(xNCHW4 + index);
                xMM = _mm_div_ps(_mm_sub_ps(xMM,meanMM),_mm_sqrt_ps(_mm_add_ps(varMM, epsMM)));

            }
        }
    }
}

void BlasNCHW4::cpuUpSampleNCHW4(float * const &inNCHW4, const int &width, const int &height, const int &channelNCHW4, const int &batch,
                                 const int &strideX, const int &strideY, const float &scale, float * const &outNCHW4)
{
    std::cout<<"not test"<<std::endl;

    __m128 scaleMM = _mm_set1_ps(scale);

    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int k = 0; k < channelNCHW4; ++k)
        {
            for (int j = 0; j < height * strideY; ++j)
            {
                for (int i = 0; i < width * strideX; ++i)
                {
                    int inIndex     =   b*width*height*channelNCHW4 + k*width*height + (j/strideY)*width + i/strideX;
                    int outIndex    =   b*width*height*channelNCHW4*strideX*strideY + k*width*height*strideX*strideY + j*width*strideX + i;

                    __m128 load     =   _mm_loadu_ps(inNCHW4+inIndex);
                    load            =   _mm_mul_ps(load, scaleMM);
                    _mm_storeu_ps(outNCHW4+outIndex, load);

                }
            }
        }
    }
}

void BlasNCHW4::cpuSoftmaxNCHW4(float * const &inputNCHW4, const int &num, const int &batch, const int &batchOff, const int &groups,
                                const int &groupOff, const float &temperature, const int &stride, float * const &outputNCHW4)
{
    std::cout<<"not test"<<std::endl;

    if( groups != 1)
    {
        throw Exception(1, "[NCHW4]: softmax multi groups not supported!", __FILE__, __LINE__, __FUNCTION__);
    }
    else
    {
        for (int b = 0; b < batch; ++b)
        {
            for (int g = 0; g < groups; ++g)
            {
                Blas::softmax(inputNCHW4 + b*batchOff + g*groupOff, num, temperature, stride,  outputNCHW4+b*batchOff+g*groupOff,true);
            }
        }
    }
}

void BlasNCHW4::cpuBilinearResizeNCHW4(float * const &inNCHW4, const int &width, const int &height, const int &channelNCHW4, const int &batch,
                                       const int &outWidth, const int &outHeight, const int &alignCorners, float * const &outNCHW4)
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
    uint64_t dataLen   = outSize;
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

        __m128 h1LamdMM    = _mm_set1_ps(h1Lamd);
        __m128 h0LamdMM    = _mm_set1_ps(h0Lamd);

        const float w1Lamd =  w1r - w1;
        const float w0Lamd =  1.0f - w1Lamd;

        __m128 w1LamdMM    = _mm_set1_ps(w1Lamd);
        __m128 w0LamdMM    = _mm_set1_ps(w0Lamd);

        const float *inPtr = inNCHW4  + (h1*width + w1)*NCHW4_PACK;
        float *outPtr      = outNCHW4 + NCHW4_PACK*i;

        for (int c = 0; c < channelNCHW4*batch; ++c)
        {
            const float* inTmp = inPtr + c*inSize*NCHW4_PACK;
            __m128 inTmpMM  = _mm_loadu_ps(inTmp);
            __m128 inTmpMM1 = _mm_loadu_ps(inTmp + w1p*NCHW4_PACK);

            __m128 inTmpMM2 = _mm_loadu_ps(inTmp + h1p*width*NCHW4_PACK);
            __m128 inTmpMM3 = _mm_loadu_ps(inTmp + (h1p*width + w1p)*NCHW4_PACK);

            __m128 part1MM = _mm_mul_ps(h0LamdMM, _mm_add_ps(_mm_mul_ps(w0LamdMM, inTmpMM) , _mm_mul_ps(w1LamdMM, inTmpMM1)));
            __m128 part2MM = _mm_mul_ps(h1LamdMM, _mm_add_ps(_mm_mul_ps(w0LamdMM, inTmpMM2), _mm_mul_ps(w1LamdMM, inTmpMM3)));

            __m128 finalMM = _mm_add_ps(part1MM,part2MM);

            _mm_storeu_ps(outPtr + c*outSize*NCHW4_PACK, finalMM);
        }

    }
}

}
