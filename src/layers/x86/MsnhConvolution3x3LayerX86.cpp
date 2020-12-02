#ifdef USE_X86

#include "Msnhnet/layers/x86/MsnhConvolution3x3LayerX86.h"

namespace Msnhnet
{

void Convolution3x3LayerX86::convolution3x3S1(float * const &src, const int &height, const int &width, const int &channel,
                                              float * &dst, const int &outHeight, const int &outWidth, const int &outChannel,
                                              float * const &kernel, const bool useFMA)
{
    int fastOutCh = (outChannel>>1)<<1;

    const int inSize        = width*height;
    const int outSize       = outWidth*outHeight;
    const int kernelSize    = 9;/* 3x3 */

    if(!useFMA)
    {

#if USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int oc = 0; oc < fastOutCh; oc+=2)
        {
            float *dst1 = dst + oc*outSize;
            float *dst2 = dst + (oc+1)*outSize;
            float *kernel1 = kernel + oc*channel*kernelSize;
            float *kernel2 = kernel + (oc+1)*channel*kernelSize;

            for (int ic = 0; ic < channel; ++ic)
            {

                float *dstPtr1 = dst1; 

                float *dstPtr2 = dst2; 

                float *dstPtr1Next = dstPtr1 + outWidth; 

                float *dstPtr2Next = dstPtr2 + outWidth; 

                float* srcNow = src + ic*inSize;

                float* line0  = srcNow;
                float* line1  = srcNow + width;
                float* line2  = srcNow + width * 2;
                float* line3  = srcNow + width * 3;

                int oh = 0;
                for (; oh < outHeight-1; oh+=2)
                {
                    for (int ow = 0; ow < outWidth; ++ow)
                    {
                        float sum1      = 0.f; 

                        float sum2      = 0.f; 

                        float sum1Next  = 0.f; 

                        float sum2Next  = 0.f; 

                        sum1 += line0[0] * kernel1[0];  

                        sum1 += line0[1] * kernel1[1];  

                        sum1 += line0[2] * kernel1[2];  

                        sum1 += line1[0] * kernel1[3];  

                        sum1 += line1[1] * kernel1[4];  

                        sum1 += line1[2] * kernel1[5];  

                        sum1 += line2[0] * kernel1[6];  

                        sum1 += line2[1] * kernel1[7];  

                        sum1 += line2[2] * kernel1[8];  

                        sum2 += line0[0] * kernel2[0];  

                        sum2 += line0[1] * kernel2[1];  

                        sum2 += line0[2] * kernel2[2];  

                        sum2 += line1[0] * kernel2[3];  

                        sum2 += line1[1] * kernel2[4];  

                        sum2 += line1[2] * kernel2[5];  

                        sum2 += line2[0] * kernel2[6];  

                        sum2 += line2[1] * kernel2[7];  

                        sum2 += line2[2] * kernel2[8];  

                        sum1Next += line1[0] * kernel1[0];   

                        sum1Next += line1[1] * kernel1[1];   

                        sum1Next += line1[2] * kernel1[2];   

                        sum1Next += line2[0] * kernel1[3];   

                        sum1Next += line2[1] * kernel1[4];   

                        sum1Next += line2[2] * kernel1[5];   

                        sum1Next += line3[0] * kernel1[6];   

                        sum1Next += line3[1] * kernel1[7];   

                        sum1Next += line3[2] * kernel1[8];   

                        sum2Next += line1[0] * kernel2[0];   

                        sum2Next += line1[1] * kernel2[1];   

                        sum2Next += line1[2] * kernel2[2];   

                        sum2Next += line2[0] * kernel2[3];   

                        sum2Next += line2[1] * kernel2[4];   

                        sum2Next += line2[2] * kernel2[5];   

                        sum2Next += line3[0] * kernel2[6];   

                        sum2Next += line3[1] * kernel2[7];   

                        sum2Next += line3[2] * kernel2[8];   

                        *(dstPtr1++)     += sum1;         

                        *(dstPtr2++)     += sum2;         

                        *(dstPtr1Next++) += sum1Next;     

                        *(dstPtr2Next++) += sum2Next;     

                        line0++;   

                        line1++;   

                        line2++;   

                        line3++;   

                    }

                    line0        += 2 + width; 

                    line1        += 2 + width; 

                    line2        += 2 + width; 

                    line3        += 2 + width; 

                    dstPtr1      += outWidth;
                    dstPtr2      += outWidth;
                    dstPtr1Next  += outWidth;
                    dstPtr2Next  += outWidth;
                }

                for (; oh < outHeight; ++oh)
                {
                    for (int ow = 0; ow < outWidth; ++ow)
                    {
                        float sum1      = 0.f; 

                        float sum2      = 0.f; 

                        sum1 += line0[0] * kernel1[0];
                        sum1 += line0[1] * kernel1[1];
                        sum1 += line0[2] * kernel1[2];
                        sum1 += line1[0] * kernel1[3];
                        sum1 += line1[1] * kernel1[4];
                        sum1 += line1[2] * kernel1[5];
                        sum1 += line2[0] * kernel1[6];
                        sum1 += line2[1] * kernel1[7];
                        sum1 += line2[2] * kernel1[8];

                        sum2 += line0[0] * kernel2[0];
                        sum2 += line0[1] * kernel2[1];
                        sum2 += line0[2] * kernel2[2];
                        sum2 += line1[0] * kernel2[3];
                        sum2 += line1[1] * kernel2[4];
                        sum2 += line1[2] * kernel2[5];
                        sum2 += line2[0] * kernel2[6];
                        sum2 += line2[1] * kernel2[7];
                        sum2 += line2[2] * kernel2[8];

                        *(dstPtr1++)     += sum1;         

                        *(dstPtr2++)     += sum2;         

                        line0++;   

                        line1++;   

                        line2++;   

                    }

                    line0        += 2;
                    line1        += 2;
                    line2        += 2;
                }

                kernel1 += kernelSize;

                kernel2 += kernelSize;

            }
        }
    }
    else
    {

#if USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int oc = 0; oc < fastOutCh; oc+=2)
        {
            float *dst1 = dst + oc*outSize;
            float *dst2 = dst + (oc+1)*outSize;

            float *kernel1 = kernel + oc*channel*kernelSize;
            float *kernel2 = kernel + (oc+1)*channel*kernelSize;

            for (int ic = 0; ic < channel; ++ic)
            {
                M128 k012;
                M128 k345;
                M128 k678;
                M128 k012Next;
                M128 k345Next;
                M128 k678Next;

                k012.m128       = _mm_loadu_ps(kernel1);
                k345.m128       = _mm_loadu_ps(kernel1+3);
                k678.m128       = _mm_loadu_ps(kernel1+6);

                k012Next.m128   = _mm_loadu_ps(kernel2);
                k345Next.m128   = _mm_loadu_ps(kernel2+3);
                k678Next.m128   = _mm_loadu_ps(kernel2+6);

                k012.f32[3]     = 0;
                k345.f32[3]     = 0;
                k678.f32[3]     = 0;

                k012Next.f32[3] = 0;
                k345Next.f32[3] = 0;
                k678Next.f32[3] = 0;

                float *dstPtr1 = dst1; 

                float *dstPtr2 = dst2; 

                float *dstPtr1Next = dstPtr1 + outWidth; 

                float *dstPtr2Next = dstPtr2 + outWidth; 

                float* srcNow = src + ic*inSize;

                float* line0  = srcNow;
                float* line1  = srcNow + width;
                float* line2  = srcNow + width * 2;
                float* line3  = srcNow + width * 3;

                int oh = 0;
                for (; oh < outHeight-1; oh+=2)
                {
                    for (int ow = 0; ow < outWidth; ++ow)
                    {
                        float sum1      = 0.f; 

                        float sum2      = 0.f; 

                        float sum1Next  = 0.f; 

                        float sum2Next  = 0.f; 

                        M128 mmSum;
                        __m128 mmLine0  = _mm_loadu_ps(line0);
                        __m128 mmLine1  = _mm_loadu_ps(line1);
                        __m128 mmLine2  = _mm_loadu_ps(line2);
                        __m128 mmLine3  = _mm_loadu_ps(line3);

                        mmSum.m128 = _mm_fmadd_ps(mmLine0,k012.m128,_mm_fmadd_ps(mmLine1,k345.m128,_mm_mul_ps(mmLine2,k678.m128)));
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        sum1       = mmSum.f32[0];

                        mmSum.m128 = _mm_fmadd_ps(mmLine0,k012Next.m128,_mm_fmadd_ps(mmLine1,k345Next.m128,_mm_mul_ps(mmLine2,k678Next.m128)));
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        sum2       = mmSum.f32[0];

                        mmSum.m128 = _mm_fmadd_ps(mmLine1,k012.m128,_mm_fmadd_ps(mmLine2,k345.m128,_mm_mul_ps(mmLine3,k678.m128)));
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        sum1Next   = mmSum.f32[0];

                        mmSum.m128 = _mm_fmadd_ps(mmLine1,k012Next.m128,_mm_fmadd_ps(mmLine2,k345Next.m128,_mm_mul_ps(mmLine3,k678Next.m128)));
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        sum2Next   = mmSum.f32[0];

                        *(dstPtr1++)     += sum1;         

                        *(dstPtr2++)     += sum2;         

                        *(dstPtr1Next++) += sum1Next;     

                        *(dstPtr2Next++) += sum2Next;     

                        line0++;   

                        line1++;   

                        line2++;   

                        line3++;   

                    }

                    line0        += 2 + width; 

                    line1        += 2 + width; 

                    line2        += 2 + width; 

                    line3        += 2 + width; 

                    dstPtr1      += outWidth;
                    dstPtr2      += outWidth;
                    dstPtr1Next  += outWidth;
                    dstPtr2Next  += outWidth;
                }

                for (; oh < outHeight; ++oh)
                {
                    for (int ow = 0; ow < outWidth; ++ow)
                    {
                        float sum1      = 0.f; 

                        float sum2      = 0.f; 

                        M128 mmSum;
                        __m128 mmLine0;
                        __m128 mmLine1;
                        __m128 mmLine2;

                        mmLine0    = _mm_loadu_ps(line0);
                        mmLine1    = _mm_loadu_ps(line1);
                        mmLine2    = _mm_loadu_ps(line2);

                        mmSum.m128 = _mm_fmadd_ps(mmLine0,k012.m128,_mm_fmadd_ps(mmLine1,k345.m128,_mm_mul_ps(mmLine2,k678.m128)));
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        sum1       = mmSum.f32[0];

                        mmSum.m128 = _mm_fmadd_ps(mmLine0,k012Next.m128,_mm_fmadd_ps(mmLine1,k345Next.m128,_mm_mul_ps(mmLine2,k678Next.m128)));
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        sum2       = mmSum.f32[0];

                        *(dstPtr1++)     += sum1;         

                        *(dstPtr2++)     += sum2;         

                        line0++;   

                        line1++;   

                        line2++;   

                    }

                    line0        += 2;
                    line1        += 2;
                    line2        += 2;
                }

                kernel1 += kernelSize;

                kernel2 += kernelSize;

            }
        }
    }

}

void Convolution3x3LayerX86::convolution3x3S2(float * const &src, const int &height, const int &width, const int &channel,
                                              float *&dst, const int &outHeight, const int &outWidth, const int &outChannel,
                                              float * const &kernel, const bool useFMA)
{
    int fastOutCh = (outChannel>>1)<<1;

    const int inSize        = width*height;
    const int outSize       = outWidth*outHeight;
    const int kernelSize    = 9;/* 3x3 */

    if(!useFMA)
    {

#if USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int oc = 0; oc < fastOutCh; oc+=2)
        {
            float *dst1 = dst + oc*outSize;
            float *dst2 = dst + (oc+1)*outSize;
            float *kernel1 = kernel + oc*channel*kernelSize;
            float *kernel2 = kernel + (oc+1)*channel*kernelSize;

            for (int ic = 0; ic < channel; ++ic)
            {

                float *dstPtr1 = dst1; 

                float *dstPtr2 = dst2; 

                float* srcNow = src + ic*inSize;

                float* line0  = srcNow;
                float* line1  = srcNow + width;
                float* line2  = srcNow + width * 2;

                int oh = 0;
                for (; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ++ow)
                    {
                        float sum1      = 0.f; 

                        float sum2      = 0.f; 

                        sum1 += line0[0] * kernel1[0];  

                        sum1 += line0[1] * kernel1[1];  

                        sum1 += line0[2] * kernel1[2];  

                        sum1 += line1[0] * kernel1[3];  

                        sum1 += line1[1] * kernel1[4];  

                        sum1 += line1[2] * kernel1[5];  

                        sum1 += line2[0] * kernel1[6];  

                        sum1 += line2[1] * kernel1[7];  

                        sum1 += line2[2] * kernel1[8];  

                        sum2 += line0[0] * kernel2[0];  

                        sum2 += line0[1] * kernel2[1];  

                        sum2 += line0[2] * kernel2[2];  

                        sum2 += line1[0] * kernel2[3];  

                        sum2 += line1[1] * kernel2[4];  

                        sum2 += line1[2] * kernel2[5];  

                        sum2 += line2[0] * kernel2[6];  

                        sum2 += line2[1] * kernel2[7];  

                        sum2 += line2[2] * kernel2[8];  

                        *(dstPtr1++)     += sum1;         

                        *(dstPtr2++)     += sum2;         

                        line0+=2;   

                        line1+=2;   

                        line2+=2;   

                    }

                    line0        += 2 * (width - outWidth); 

                    line1        += 2 * (width - outWidth); 

                    line2        += 2 * (width - outWidth); 

                }
                kernel1 += kernelSize;

                kernel2 += kernelSize;

            }
        }

#if USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int oc = fastOutCh; oc < outChannel; oc++)
        {
            float *dst1 = dst + oc*outSize;
            float *kernel1 = kernel + oc*channel*kernelSize;

            for (int ic = 0; ic < channel; ++ic)
            {

                float *dstPtr1 = dst1; 

                float* srcNow = src + ic*inSize;

                float* line0  = srcNow;
                float* line1  = srcNow + width;
                float* line2  = srcNow + width * 2;

                int oh = 0;
                for (; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ++ow)
                    {
                        float sum1      = 0.f; 

                        sum1 += line0[0] * kernel1[0];  

                        sum1 += line0[1] * kernel1[1];  

                        sum1 += line0[2] * kernel1[2];  

                        sum1 += line1[0] * kernel1[3];  

                        sum1 += line1[1] * kernel1[4];  

                        sum1 += line1[2] * kernel1[5];  

                        sum1 += line2[0] * kernel1[6];  

                        sum1 += line2[1] * kernel1[7];  

                        sum1 += line2[2] * kernel1[8];  

                        *(dstPtr1++)     += sum1;         

                        line0+=2;   

                        line1+=2;   

                        line2+=2;   

                    }

                    line0        += 2 * (width - outWidth); 

                    line1        += 2 * (width - outWidth); 

                    line2        += 2 * (width - outWidth); 

                }
                kernel1 += kernelSize;

            }
        }
    }
    else
    {

#if USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int oc = 0; oc < fastOutCh; oc+=2)
        {
            float *dst1 = dst + oc*outSize;
            float *dst2 = dst + (oc+1)*outSize;
            float *kernel1 = kernel + oc*channel*kernelSize;
            float *kernel2 = kernel + (oc+1)*channel*kernelSize;

            for (int ic = 0; ic < channel; ++ic)
            {
                M128 k012;
                M128 k345;
                M128 k678;
                M128 k012Next;
                M128 k345Next;
                M128 k678Next;

                k012.m128       = _mm_loadu_ps(kernel1);
                k345.m128       = _mm_loadu_ps(kernel1+3);
                k678.m128       = _mm_loadu_ps(kernel1+6);

                k012Next.m128   = _mm_loadu_ps(kernel2);
                k345Next.m128   = _mm_loadu_ps(kernel2+3);
                k678Next.m128   = _mm_loadu_ps(kernel2+6);

                k012.f32[3]     = 0;
                k345.f32[3]     = 0;
                k678.f32[3]     = 0;

                k012Next.f32[3] = 0;
                k345Next.f32[3] = 0;
                k678Next.f32[3] = 0;

                float *dstPtr1 = dst1; 

                float *dstPtr2 = dst2; 

                float* srcNow = src + ic*inSize;

                float* line0  = srcNow;
                float* line1  = srcNow + width;
                float* line2  = srcNow + width * 2;

                int oh = 0;
                for (; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ++ow)
                    {
                        float sum1      = 0.f; 

                        float sum2      = 0.f; 

                        M128 mmSum;
                        __m128 mmLine0  = _mm_loadu_ps(line0);
                        __m128 mmLine1  = _mm_loadu_ps(line1);
                        __m128 mmLine2  = _mm_loadu_ps(line2);

                        mmSum.m128 = _mm_fmadd_ps(mmLine0,k012.m128,_mm_fmadd_ps(mmLine1,k345.m128,_mm_mul_ps(mmLine2,k678.m128)));
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        sum1       = mmSum.f32[0];

                        mmSum.m128 = _mm_fmadd_ps(mmLine0,k012Next.m128,_mm_fmadd_ps(mmLine1,k345Next.m128,_mm_mul_ps(mmLine2,k678Next.m128)));
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        sum2       = mmSum.f32[0];

                        *(dstPtr1++)     += sum1;         

                        *(dstPtr2++)     += sum2;         

                        line0+=2;   

                        line1+=2;   

                        line2+=2;   

                    }

                    line0        += 2 * (width - outWidth); 

                    line1        += 2 * (width - outWidth); 

                    line2        += 2 * (width - outWidth); 

                }
                kernel1 += kernelSize;

                kernel2 += kernelSize;

            }
        }

#if USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int oc = fastOutCh; oc < outChannel; oc++)
        {
            float *dst1 = dst + oc*outSize;
            float *kernel1 = kernel + oc*channel*kernelSize;

            for (int ic = 0; ic < channel; ++ic)
            {

                M128 k012;
                M128 k345;
                M128 k678;

                k012.m128       = _mm_loadu_ps(kernel1);
                k345.m128       = _mm_loadu_ps(kernel1+3);
                k678.m128       = _mm_loadu_ps(kernel1+6);

                k012.f32[3]     = 0;
                k345.f32[3]     = 0;
                k678.f32[3]     = 0;

                float *dstPtr1 = dst1; 

                float* srcNow = src + ic*inSize;

                float* line0  = srcNow;
                float* line1  = srcNow + width;
                float* line2  = srcNow + width * 2;

                int oh = 0;
                for (; oh < outHeight; oh++)
                {
                    for (int ow = 0; ow < outWidth; ++ow)
                    {
                        float sum1      = 0.f; 

                        M128 mmSum;
                        __m128 mmLine0  = _mm_loadu_ps(line0);
                        __m128 mmLine1  = _mm_loadu_ps(line1);
                        __m128 mmLine2  = _mm_loadu_ps(line2);

                        mmSum.m128 = _mm_fmadd_ps(mmLine0,k012.m128,_mm_fmadd_ps(mmLine1,k345.m128,_mm_mul_ps(mmLine2,k678.m128)));
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        mmSum.m128 = _mm_hadd_ps(mmSum.m128,mmSum.m128);
                        sum1       = mmSum.f32[0];
                        *(dstPtr1++)     += sum1;         

                        line0+=2;   

                        line1+=2;   

                        line2+=2;   

                    }

                    line0        += 2 * (width - outWidth); 

                    line1        += 2 * (width - outWidth); 

                    line2        += 2 * (width - outWidth); 

                }
                kernel1 += kernelSize;

            }
        }
    }

}

}

#endif
