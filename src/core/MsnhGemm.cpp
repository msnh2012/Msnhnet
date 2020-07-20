#include "Msnhnet/core/MsnhGemm.h"
namespace Msnhnet
{
uint8_t Gemm::lookup[16] = { 0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,};

void Gemm::repackInput(float * const &input, float * const &rePackedInput, const int &width, const int &height, const int &channel)
{
    const int itemsPerCh    =  width * height;

    for(int ch =0; ch < channel; ch+=32)
    {
        for (int i = 0; i < itemsPerCh; ++i)
        {
            for (int chPack = 0; chPack < 32; ++chPack)
            {
                float src = input[(ch + chPack)*itemsPerCh + i];
                rePackedInput[ch * itemsPerCh + i*32 + chPack] = src;
            }
        }
    }
}

void Gemm::float2Bit(float * const &input, uint8_t * const &output, size_t size)
{
#ifdef USE_X86
    size_t outSize  =  size / 8 + 1;

    memset(output ,0, outSize);

    __m256 floatZero256 = _mm256_set1_ps(0.0);

    for (size_t i = 0; i < size; ++i)
    {
        __m256 src256   =  _mm256_loadu_ps(static_cast<float*>((&input[i])));
        __m256 resut256 =  _mm256_cmp_ps(src256, floatZero256, _CMP_GT_OS);
        uint32_t mask   =  static_cast<uint32_t>(_mm256_movemask_ps(resut256));

        output[i / 8]   = static_cast<uint8_t>(mask);
    }
#endif

#ifdef USE_ARM
    (void)input;
    (void)output;
    (void)size;
    throw Exception(1,"TODO: for arm",__FILE__, __LINE__);
#endif
}

void Gemm::cpuIm2col(float * const &input, const int &channelNum, const int &height, const int &width,
                     const int &kSize, const int &stride, const int &padding, float * const &output)
{
    const int heightCol  = (height + 2*padding - kSize) / stride + 1;
    const int widthCol   = (width  + 2*padding - kSize) / stride + 1;

    const int chCols     = channelNum * kSize * kSize;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int ch = 0; ch < chCols; ++ch)
    {

        int wOffset = ch % kSize;
        int hOffset = (ch / kSize) % kSize;
        int chOff   = ch / kSize / kSize;

        for (int h = 0; h < heightCol; ++h)
        {
            for (int w = 0; w < widthCol; ++w)
            {

                int imRow           = hOffset + h*stride;
                int imCol           = wOffset + w*stride;

                int colIndex        = (ch*heightCol + h)*widthCol + w;

                output[colIndex]    = img2ColGetPixel(input, height, width, imRow, imCol, chOff, padding);

            }
        }
    }
}

void Gemm::cpuCol2Im(float * const &input, const int &channelNum, const int &height, const int &width, const int &kSizeX, const int &kSizeY,
                      const int &strideX, const int &strideY, const int &paddingX, const int &paddingY, float * const &output)
{
    int heightCol   =   (height + 2*paddingY - kSizeY)/strideY + 1;
    int widthCol    =   (width  + 2*paddingX - kSizeX)/strideX + 1;

    int channelsCol =   channelNum * kSizeX * kSizeY;

    for (int c = 0; c < channelsCol; ++c)
    {
        int wOffset =   c % kSizeX;
        int hOffset =   (c / kSizeX ) % kSizeY;

        int cIm     =   c / kSizeX / kSizeY;

        for (int h = 0; h < heightCol; ++h)
        {
            for (int w = 0; w < widthCol; ++w)
            {
                int imRow       =   hOffset + h * strideY;
                int imCol       =   wOffset + w * strideX;

                int colIndex    =   (c * heightCol + h) * widthCol + w;
                float val       =   input[colIndex];

                imRow           =   imRow - paddingY;
                imCol           =   imCol - paddingX;

                if(imRow < 0 || imCol <0 || imRow >= height|| imCol >= width)
                {

                }
                else
                {
                    output[imCol + width*(imRow + height*cIm)] += val;
                }
            }
        }
    }
}

void Gemm::cpuIm2colEx(float *input, const int &channelNum, const int &height, const int &width,
                       const int &kernelH, const int &kernelW, const int &padH, const int &padW,
                       const int &strideH, const int &strideW, const int &dilationH, const int &dilationW,
                       float *output)
{

    const int outputH       =   (height + 2 * padH - (dilationH * (kernelH - 1) + 1)) / strideH + 1;
    const int outputW       =   (width  + 2 * padW - (dilationW * (kernelW - 1) + 1)) / strideW + 1;

    const int channelSize   =   height * width;

    if(outputH == height && outputW == width && strideH==1 &&strideW==1 && padH==1 && padW==1)
    {
#ifdef X86
        cpuIm2colWithAvx(input, channelNum, height, width, kernelH, strideH, padH, output, 1);
#else
       goto NEXT;
#endif
    }
    else
    {
NEXT:        
        for (int channel = 0 ; channel++<channelNum; input += channelSize) 
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int kernelRow = 0; kernelRow < kernelH; kernelRow++)   
            {
                for (int kernelCol = 0; kernelCol < kernelW; kernelCol++) 
                {

                    int inputRow = -padH + kernelRow * dilationH;

                    for (int outputRow = 0; outputRow < outputH; ++outputRow)
                    {
                        if (!is_a_ge_zero_and_a_lt_b(inputRow + outputRow*strideH, height)) 
                        {
                            for (int outputCol = 0; outputCol < outputW; ++outputCol) 
                            {
                                output[ (channel-1)*kernelH*kernelW*outputH*outputW + kernelRow *kernelW*outputH*outputW +
                                        kernelCol*outputH*outputW + outputRow*outputW + outputCol] = 0;
                            }
                        }
                        else
                        {
                            int inputCol = -padW + kernelCol * dilationW; 

                            for (int outputCol = 0 ; outputCol<outputW; ++outputCol)
                            {
                                if (is_a_ge_zero_and_a_lt_b(inputCol + strideW*outputCol, width)) 
                                {
                                    output[ (channel-1)*kernelH*kernelW*outputH*outputW + kernelRow *kernelW*outputH*outputW +
                                            kernelCol*outputH*outputW + outputRow*outputW + outputCol]
                                            = input[(inputRow + outputRow*strideH) * width + inputCol + strideW*outputCol]; 
                                }
                                else    
                                {
                                    output[(channel-1)*kernelH*kernelW*outputH*outputW + kernelRow *kernelW*outputH*outputW +
                                            kernelCol*outputH*outputW + outputRow*outputW + outputCol] = 0; 
                                }
                            }
                        }
                    }

                }
            }
        }
    }
}

void Gemm::cpuCol2ImEx(float *input, const int &channelNum, const int &height, const int &width, const int &kernelH, const int &kernelW, const int &padH, const int &padW, const int &strideH, const int &strideW, const int &dilationH, const int &dilationW, float *output)
{
    const int outputH   =   (height + 2*padH - (dilationH * (kernelH - 1) + 1))/strideH + 1;
    const int outputW   =   (width  + 2*padW - (dilationW * (kernelW - 1) + 1))/strideW + 1;

    const int channelSize   =   height * width;

    for (int channel = 0 ; channel++<channelNum; input += channelSize) 
    {

        for (int kernelRow = 0; kernelRow < kernelH; kernelRow++)   
        {
            for (int kernelCol = 0; kernelCol < kernelW; kernelCol++) 
            {

                int inputRow = -padH + kernelRow * dilationH;

                for (int outputRow = 0; outputRow < outputH; ++outputRow)
                {
                    if (!is_a_ge_zero_and_a_lt_b(inputRow + outputRow*strideH, height)) 
                    {
                        input += outputW;

                    }
                    else
                    {
                        int inputCol = -padW + kernelCol * dilationW; 

                        for (int outputCol = 0 ; outputCol<outputW; ++outputCol)
                        {
                            if (is_a_ge_zero_and_a_lt_b(inputCol + strideW*outputCol, width)) 
                            {
                                output[ (channel-1)*kernelH*kernelW*outputH*outputW + kernelRow *kernelW*outputH*outputW +
                                        kernelCol*outputH*outputW + outputRow*outputW + outputCol]
                                        = *input;
                            }

                            input++;

                        }
                    }
                }

            }
        }
    }
}

void Gemm::cpuIm2colWithAvx(float * const &input, const int &channelNum, const int &height, const int &width, const int &kSize,
                            const int &stride, const int &padding, float * const &output, const bool &supportAvxAndFma)
{

#ifdef USE_X86
    if(supportAvxAndFma)
    {
        const int heightCol  = (height + 2*padding - kSize) / stride + 1;
        const int widthCol   = (width  + 2*padding - kSize) / stride + 1;
        const int chCols     = channelNum * kSize * kSize;

        if(heightCol == height && widthCol == width && stride == 1 && padding == 1)
        {
#pragma omp parallel for num_threads(OMP_THREAD)
            for (int ch = 0; ch < chCols; ++ch)
            {
                int h       = 0;
                int w       = 0;
                int wOffset = ch % kSize;
                int hOffset = (ch / kSize) % kSize;
                int chOff   = ch / kSize / kSize;

                for (h = padding; h < heightCol - padding; ++h)
                {
                    for (w = padding; w < widthCol - padding - 8; w+=8)
                    {

                        int imRow           = hOffset + h - padding;
                        int imCol           = wOffset + w - padding;

                        int colIndex        = (ch*heightCol + h)*widthCol + w;

                        __m256 src256       = _mm256_loadu_ps(static_cast<float*>((&input[imCol + width*(imRow + heightCol * chOff)])));

                        _mm256_storeu_ps(&output[colIndex], src256);
                    }

                    for (; w < widthCol - padding; ++w)
                    {
                        int imRow           = hOffset + h - padding;
                        int imCol           = wOffset + w - padding;
                        int colIndex        = (ch*heightCol + h)*widthCol + w;

                        output[colIndex]    = input[imCol + width*(imRow + heightCol * chOff)];
                    }
                }

                {   
                    w = 0;
                    for (h = 0; h < heightCol; ++h)
                    {

                        int imRow           = hOffset + h*stride;
                        int imCol           = wOffset + w*stride;

                        int colIndex        = (ch*heightCol + h)*widthCol + w;

                        output[colIndex]    = img2ColGetPixel(input, height, width, imRow, imCol, chOff, padding);
                    }
                }

                {   
                    w = widthCol - 1;
                    for (h = 0; h < heightCol; ++h)
                    {

                        int imRow           = hOffset + h*stride;
                        int imCol           = wOffset + w*stride;

                        int colIndex        = (ch*heightCol + h)*widthCol + w;

                        output[colIndex]    = img2ColGetPixel(input, height, width, imRow, imCol, chOff, padding);
                    }
                }

                {

                    h = 0;
                    for (w = 0; w < widthCol; ++w)
                    {

                        int imRow           = hOffset + h*stride;
                        int imCol           = wOffset + w*stride;

                        int colIndex        = (ch*heightCol + h)*widthCol + w;

                        output[colIndex]    = img2ColGetPixel(input, height, width, imRow, imCol, chOff, padding);
                    }
                }

                {

                    h = heightCol - 1;
                    for (w = 0; w < widthCol; ++w)
                    {

                        int imRow           = hOffset + h*stride;
                        int imCol           = wOffset + w*stride;

                        int colIndex        = (ch*heightCol + h)*widthCol + w;

                        output[colIndex]    = img2ColGetPixel(input, height, width, imRow, imCol, chOff, padding);
                    }
                }

            }
        }
    }
    else
    {
        cpuIm2col(input,channelNum,height,width,kSize,stride,padding,output);
    }
#endif

#ifdef USE_ARM
    (void)supportAvxAndFma;
    cpuIm2col(input,channelNum,height,width,kSize,stride,padding,output);
#endif
}

void Gemm::cpuIm2colBinWithAvx(float * const &input, const int &channelNum, const int &height, const int &width,
                               const int &kSize, const int &stride, const int &padding, float * const &output,
                               const int &bitAlign, const bool &supportAvxAndFma)
{
#ifdef USE_X86
    if(supportAvxAndFma)
    {
        const int heightCol  = (height + 2*padding - kSize) / stride + 1;
        const int widthCol   = (width  + 2*padding - kSize) / stride + 1;
        const int chCols     = channelNum * kSize * kSize;

        if(heightCol == height && widthCol == width && stride == 1 && padding == 1)
        {

            __m256i all256Single1 = _mm256_set_epi32(static_cast<int>(0x80000000), static_cast<int>(0x80000000), static_cast<int>(0x80000000), static_cast<int>(0x80000000),
                                                     static_cast<int>(0x80000000), static_cast<int>(0x80000000), static_cast<int>(0x80000000), static_cast<int>(0x80000000));
            __m256  floatZero256  = _mm256_set1_ps(0.00);
            int newLdb            = bitAlign;

#pragma omp parallel for num_threads(OMP_THREAD)
            for (int ch = 0; ch < chCols; ++ch)
            {
                int h       = 0;
                int w       = 0;
                int wOffset = ch % kSize;
                int hOffset = (ch / kSize) % kSize;
                int chOff   = ch / kSize / kSize;

                for (h = padding; h < heightCol - padding; ++h)
                {
                    for (w = padding; w < widthCol - padding - 8; w+=8)
                    {

                        int imRow           = hOffset + h - padding;
                        int imCol           = wOffset + w - padding;

                        int colIndex        = ch*newLdb + h*widthCol + w;

                        __m256 src256       = _mm256_loadu_ps(static_cast<float*>((&input[imCol + width*(imRow + heightCol * chOff)])));
                        __m256 result256    = _mm256_cmp_ps(src256, floatZero256, _CMP_GT_OS);
                        uint16_t mask       = _mm256_movemask_ps(result256); 

                        uint16_t* dstPtr = (uint16_t*)&((uint8_t*)output)[colIndex / 8];
                        *dstPtr |= (mask << (colIndex % 8));
                    }

                    for (; w < widthCol - padding; ++w)
                    {
                        int imRow           = hOffset + h - padding;
                        int imCol           = wOffset + w - padding;
                        int colIndex        = ch*newLdb + h*widthCol + w;

                        float value         = input[imCol + width*(imRow + height*chOff)];
                        if(value>0)
                        {

                            setBit((uint8_t*)output,colIndex);
                        }
                    }
                }

                {   
                    w = 0;
                    for (h = 0; h < heightCol; ++h)
                    {

                        int imRow           = hOffset + h*stride;
                        int imCol           = wOffset + w*stride;

                        int colIndex        = ch*newLdb + h*widthCol + w;

                        float value          = img2ColGetPixel(input, height, width, imRow, imCol, chOff, padding);

                        if(value>0)
                        {
                            setBit((uint8_t*)output,colIndex);
                        }
                    }
                }

                {   
                    w = widthCol - 1;
                    for (h = 0; h < heightCol; ++h)
                    {

                        int imRow           = hOffset + h*stride;
                        int imCol           = wOffset + w*stride;

                        int colIndex        = ch*newLdb + h*widthCol + w;

                        float value         = img2ColGetPixel(input, height, width, imRow, imCol, chOff, padding);

                        if(value>0)
                        {
                            setBit((uint8_t*)output,colIndex);
                        }
                    }
                }

                {

                    h = 0;
                    for (w = 0; w < widthCol; ++w)
                    {

                        int imRow           = hOffset + h*stride;
                        int imCol           = wOffset + w*stride;

                        int colIndex        = ch*newLdb + h*widthCol + w;

                        float value         = img2ColGetPixel(input, height, width, imRow, imCol, chOff, padding);

                        if(value>0)
                        {
                            setBit((uint8_t*)output,colIndex);
                        }
                    }
                }

                {

                    h = heightCol - 1;
                    for (w = 0; w < widthCol; ++w)
                    {

                        int imRow           = hOffset + h*stride;
                        int imCol           = wOffset + w*stride;

                        int colIndex        = ch*newLdb + h*widthCol + w;

                        float value         = img2ColGetPixel(input, height, width, imRow, imCol, chOff, padding);

                        if(value>0)
                        {
                            setBit((uint8_t*)output,colIndex);
                        }
                    }
                }
            }
        }

    }
    else
    {
        throw Exception(1,"Error: is no non-optimized version",__FILE__, __LINE__);
    }
#endif

#ifdef USE_ARM

    (void)input;
    (void)channelNum;
    (void)height;
    (void)width;
    (void)kSize;
    (void)stride;
    (void)padding;
    (void)output;
    (void)bitAlign;
    (void)supportAvxAndFma;
    throw Exception(1,"Error: is no non-optimized version",__FILE__, __LINE__);
#endif
}

float Gemm::img2ColGetPixel(float * const &input, const int &height, const int &width, const int &row,
                            const int &col, const int &channel, const int &padding)
{
    int mRow    =  row - padding;
    int mCol    =  col - padding;
    if(mRow < 0 || mCol < 0 || mRow >=height || mCol >=width) 
    {
        return 0;
    }

    return input[mCol + width*(mRow + height*channel)];

}

void Gemm::cpuGemm(const int &TA, const int &TB, const int &M, const int &N, const int &K, const float &ALPHA, float * const &A, const int &lda, float * const &B, const int &ldb, const float &BETA, float * const &C, const int &ldc, const bool &supportAvxAndFma)
{

#ifdef USE_OPEN_BLAS

    cblas_sgemm(CblasRowMajor,TA==1?CblasTrans:CblasNoTrans, TB==1?CblasTrans:CblasNoTrans,M,N,K,ALPHA,A,lda,B,ldb,BETA,C,ldc);

#else

    if(BETA!=1.f)
    {

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                C[i*ldc + j] *= BETA;
            }
        }
    }

#ifdef  USE_X86

    if(supportAvxAndFma && TA!=1 && TB!=1)
    {
        cpuGemmNNFast(M,N,K,ALPHA,A,lda,B,ldb,C,ldc);
    }
    else
    {
#pragma omp parallel for num_threads(OMP_THREAD)
        for (int m = 0; m < M; ++m)
        {
            if(TA!=1 && TB!=1)
            {

                cpuGemmNN(1,N,K,ALPHA,A+lda*m, lda, B, ldb, C+m*ldc, ldc,supportAvxAndFma);
            }
            else if(TA==1 && TB!=1)
            {

                cpuGemmTN(1,N,K,ALPHA,A+m, lda, B, ldb, C+m*ldc, ldc,supportAvxAndFma);
            }
            else if(TA!=1 && TB ==1)
            {

                cpuGemmNT(1,N,K,ALPHA,A+lda*m, lda, B, ldb, C+m*ldc, ldc,supportAvxAndFma);
            }
            else
            {

                cpuGemmTT(1,N,K,ALPHA,A+m, lda, B, ldb, C+m*ldc, ldc,supportAvxAndFma);
            }
        }
    }
#endif
#endif

#ifdef USE_ARM
#ifndef USE_OPEN_BLAS
    (void)supportAvxAndFma;
#pragma omp parallel for num_threads(OMP_THREAD)
    for (int m = 0; m < M; ++m)
    {
        if(TA!=1 && TB!=1)
        {

            cpuGemmNN(1,N,K,ALPHA,A+lda*m, lda, B, ldb, C+m*ldc, ldc,false);
        }
        else if(TA==1 && TB!=1)
        {

            cpuGemmTN(1,N,K,ALPHA,A+m, lda, B, ldb, C+m*ldc, ldc,false);
        }
        else if(TA!=1 && TB ==1)
        {

            cpuGemmNT(1,N,K,ALPHA,A+lda*m, lda, B, ldb, C+m*ldc, ldc,false);
        }
        else
        {

            cpuGemmTT(1,N,K,ALPHA,A+m, lda, B, ldb, C+m*ldc, ldc,false);
        }
    }
#endif
#endif
}

void Gemm::cpuGemmNN(const int &M, const int &N, const int &K, const float &ALPHA,
                     float * const &A, const int &lda, 
                     float * const &B, const int &ldb, 
                     float * const &C, const int &ldc, 
                     const bool &supportAvxAndFma)
{

#ifdef USE_X86
    if(supportAvxAndFma)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < M; ++i)         
        {
            for (int k = 0; k < K; ++k)     
            {
                __m256 a256, b256, c256, result256;    
                float A_PART =  ALPHA*A[i*lda + k];     

                a256         =  _mm256_set1_ps(A_PART);
                for (int j = 0; j < N - 8; j += 8)     
                {
                    b256 = _mm256_loadu_ps(&B[k*ldb + j]); 
                    c256 = _mm256_loadu_ps(&C[i*ldc + j]); 

                    result256 = _mm256_mul_ps(a256, b256);     
                    result256 = _mm256_add_ps(result256, c256);
                    _mm256_storeu_ps(&C[i*ldc + j], result256);
                }

                int prevEnd = (N % 8 == 0) ? (N - 8) : (N / 8) * 8; 

                for (int j = prevEnd; j < N; ++j)   
                {
                    C[i*ldc + j] += A_PART*B[k*ldb + j];
                }
            }
        }
    }
    else
    {

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < M; ++i)   
        {
            for (int k = 0; k < K; ++k)     
            {
                float A_PART =  ALPHA*A[i*lda + k];     

                for (int j = 0; j < N; ++j)  
                {
                    C[i*ldc + j] += A_PART*B[k*ldb + j];
                }
            }
        }
    }
#endif

#ifdef USE_ARM

#ifdef USE_NEON
    (void) supportAvxAndFma;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < M; ++i)   
    {
        for (int k = 0; k < K; ++k)     
        {
            float32x4_t a, b, c, result;

            float A_PART = ALPHA*A[i*lda+k];

            a = vdupq_n_f32(A_PART);

            for(int j = 0; j < N-4; j+=4)
            {
                b = vld1q_f32(&B[k*ldb + j]);
                c = vld1q_f32(&C[i*ldc + j]);

                result = vmulq_f32(a,b);
                result = vaddq_f32(result, c);

                vst1q_f32(&C[i*ldc + j],result);
            }

            int prevEnd = (N % 4 == 0) ? (N - 4) : (N / 4) * 4; 

            for (int j = prevEnd; j < N; ++j)   
            {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }

#else
    (void) supportAvxAndFma;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < M; ++i)   
    {
        for (int k = 0; k < K; ++k)     
        {
            float A_PART =  ALPHA*A[i*lda + k];     

            for (int j = 0; j < N; ++j)  
            {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
#endif

#endif
}

void Gemm::cpuGemmTN(const int &M, const int &N, const int &K, const float &ALPHA,
                     float * const &A, const int &lda,
                     float * const &B, const int &ldb,
                     float * const &C, const int &ldc,
                     const bool &supportAvxAndFma)
{

    (void)supportAvxAndFma;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < M; ++i)   
    {
        for (int k = 0; k < K; ++k)     
        {
            float A_PART =  ALPHA*A[k*lda + i];     

            for (int j = 0; j < N; ++j)  
            {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
}

void Gemm::cpuGemmNT(const int &M, const int &N, const int &K, const float &ALPHA,
                     float * const &A, const int &lda,
                     float * const &B, const int &ldb,
                     float * const &C, const int &ldc,
                     const bool &supportAvxAndFma)
{

    (void)supportAvxAndFma;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0;

            for (int k = 0; k < K; ++k)
            {
                sum += ALPHA*A[i*lda + k]*B[j*ldb + k];
            }

            C[i*ldc + j] += sum;
        }
    }
}

void Gemm::cpuGemmTT(const int &M, const int &N, const int &K, const float &ALPHA,
                     float * const &A, const int &lda,
                     float * const &B, const int &ldb,
                     float * const &C, const int &ldc,
                     const bool &supportAvxAndFma)
{

    (void)supportAvxAndFma;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < M; ++i)
    {

        for (int j = 0; j < N; ++j)
        {
            float sum = 0;
            for (int k = 0; k < K; ++k)
            {
                sum += ALPHA*A[i + k*lda ]*B[k + j*ldb];
            }

            C[i*ldc + j] += sum;
        }
    }
}

void Gemm::cpuFastADotB(const int &n, float * const &A, float * const &B, float *const &C)
{
#ifdef USE_X86
    int ptr  =   0;
    for (int i = 0; i < n/8; ++i)
    {
        ptr      =  i*8;
        __m256 a = _mm256_loadu_ps(A);
        __m256 b = _mm256_loadu_ps(B);

        __m256 c = _mm256_mul_ps(a,b);

        _mm256_store_ps(C,c);
    }

    for (int i = ptr ; i < n - ptr; ++i)
    {
        C[i]    =   A[i] * B[i];
    }
#endif

#ifdef USE_ARM
    (void)n;
    (void)A;
    (void)B;
    (void)C;
    throw Exception(1, "TODO: for arm", __FILE__, __LINE__);
#endif
}

void Gemm::cpuGemmNNFast(const int &M, const int &N, const int &K, const float &ALPHA,
                         float * const &A, const int &lda,
                         float * const &B, const int &ldb,
                         float * const &C, const int &ldc)
{
#ifdef USE_X86

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < (M / TILE_M)*TILE_M; i += TILE_M)
    {
        for (int k = 0; k < (K / TILE_K)*TILE_K; k += TILE_K)
        {
            for (int j = 0; j < (N / TILE_N)*TILE_N; j += TILE_N)
            {

                __m256 result256;
                __m256 a256_0, b256_0;    
                __m256 a256_1, b256_1;    
                __m256 a256_2;
                __m256 a256_3;
                __m256 c256_0, c256_1, c256_2, c256_3;
                __m256 c256_4, c256_5, c256_6, c256_7;

                c256_0 = _mm256_loadu_ps(&C[(0 + i)*ldc + (0 + j)]);
                c256_1 = _mm256_loadu_ps(&C[(1 + i)*ldc + (0 + j)]);
                c256_2 = _mm256_loadu_ps(&C[(0 + i)*ldc + (8 + j)]);
                c256_3 = _mm256_loadu_ps(&C[(1 + i)*ldc + (8 + j)]);

                c256_4 = _mm256_loadu_ps(&C[(2 + i)*ldc + (0 + j)]);
                c256_5 = _mm256_loadu_ps(&C[(3 + i)*ldc + (0 + j)]);
                c256_6 = _mm256_loadu_ps(&C[(2 + i)*ldc + (8 + j)]);
                c256_7 = _mm256_loadu_ps(&C[(3 + i)*ldc + (8 + j)]);

                for (int k_d = 0; k_d < (TILE_K); ++k_d)
                {
                    a256_0 = _mm256_set1_ps(ALPHA*A[(0 + i)*lda + (k_d + k)]);
                    a256_1 = _mm256_set1_ps(ALPHA*A[(1 + i)*lda + (k_d + k)]);

                    a256_2 = _mm256_set1_ps(ALPHA*A[(2 + i)*lda + (k_d + k)]);
                    a256_3 = _mm256_set1_ps(ALPHA*A[(3 + i)*lda + (k_d + k)]);

                    b256_0 = _mm256_loadu_ps(&B[(k_d + k)*ldb + (0 + j)]);
                    b256_1 = _mm256_loadu_ps(&B[(k_d + k)*ldb + (8 + j)]);

                    result256 = _mm256_mul_ps(a256_0, b256_0);
                    c256_0 = _mm256_add_ps(result256, c256_0);

                    result256 = _mm256_mul_ps(a256_1, b256_0);
                    c256_1 = _mm256_add_ps(result256, c256_1);

                    result256 = _mm256_mul_ps(a256_0, b256_1);
                    c256_2 = _mm256_add_ps(result256, c256_2);

                    result256 = _mm256_mul_ps(a256_1, b256_1);
                    c256_3 = _mm256_add_ps(result256, c256_3);

                    result256 = _mm256_mul_ps(a256_2, b256_0);
                    c256_4 = _mm256_add_ps(result256, c256_4);

                    result256 = _mm256_mul_ps(a256_3, b256_0);
                    c256_5 = _mm256_add_ps(result256, c256_5);

                    result256 = _mm256_mul_ps(a256_2, b256_1);
                    c256_6 = _mm256_add_ps(result256, c256_6);

                    result256 = _mm256_mul_ps(a256_3, b256_1);
                    c256_7 = _mm256_add_ps(result256, c256_7);
                }
                _mm256_storeu_ps(&C[(0 + i)*ldc + (0 + j)], c256_0);
                _mm256_storeu_ps(&C[(1 + i)*ldc + (0 + j)], c256_1);
                _mm256_storeu_ps(&C[(0 + i)*ldc + (8 + j)], c256_2);
                _mm256_storeu_ps(&C[(1 + i)*ldc + (8 + j)], c256_3);

                _mm256_storeu_ps(&C[(2 + i)*ldc + (0 + j)], c256_4);
                _mm256_storeu_ps(&C[(3 + i)*ldc + (0 + j)], c256_5);
                _mm256_storeu_ps(&C[(2 + i)*ldc + (8 + j)], c256_6);
                _mm256_storeu_ps(&C[(3 + i)*ldc + (8 + j)], c256_7);
            }

            for (int j = (N / TILE_N)*TILE_N; j < N; ++j)
            {
                for (int i_d = i; i_d < (i + TILE_M); ++i_d)
                {
                    for (int k_d = k; k_d < (k + TILE_K); ++k_d)
                    {
                        float A_PART = ALPHA*A[i_d*lda + k_d];
                        C[i_d*ldc + j] += A_PART*B[k_d*ldb + j];
                    }
                }
            }
        }

        for (int k = (K / TILE_K)*TILE_K; k < K; ++k)
        {
            for (int i_d = i; i_d < (i + TILE_M); ++i_d)
            {
                float A_PART = ALPHA*A[i_d*lda + k];
                for (int j = 0; j < N; ++j)
                {
                    C[i_d*ldc + j] += A_PART*B[k*ldb + j];
                }
            }
        }
    }

    for (int i = (M / TILE_M)*TILE_M; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            float A_PART = ALPHA*A[i*lda + k];
            for (int j = 0; j < N; ++j)
            {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
#endif

#ifdef USE_ARM
    (void)M;
    (void)N;
    (void)K;
    (void)ALPHA;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    throw Exception(1, "TODO: for arm", __FILE__, __LINE__);
#endif
}

void Gemm::swapVal(uint32_t &a0, uint32_t &a1, int &j, unsigned &m)
{
    uint32_t t = 0;
    t = (a0 ^ (a1 >>j)) & m;
    a0 = a0 ^ t;
    a1 = a1 ^ (t << j);
}

void Gemm::transpose32Optimized(uint32_t * const &A, const int &num)
{
    unsigned m = 0;
    int j      = 0;

    if(num!=32)
    {
        throw Exception(1, "num must equa 32", __FILE__, __LINE__);
    }

    j = 16;
    m = 0x0000FFFF;
    for (int k = 0; k < 32; k = (k + j + 1) & ~j)
    {
        swapVal(A[k], A[k + j], j, m);
    }

    j = 8;
    m = 0x00ff00ff;
    for (int k = 0; k < 32; k = (k + j + 1) & ~j)
    {
        swapVal(A[k], A[k + j], j, m);
    }

    j = 4;
    m = 0x0f0f0f0f;
    for (int k = 0; k < 32; k = (k + j + 1) & ~j)
    {
        swapVal(A[k], A[k + j], j, m);
    }

    j = 2;
    m = 0x33333333;
    for (int k = 0; k < 32; k = (k + j + 1) & ~j)
    {
        swapVal(A[k], A[k + j], j, m);
    }

    j = 1;
    m = 0x55555555;
    for (int k = 0; k < 32; k = (k + j + 1) & ~j)
    {
        swapVal(A[k], A[k + j], j, m);
    }

    for (j = 0; j < 16; ++j)
    {
        uint32_t tmp = A[j];
        A[j]      = reverse32Bit(A[31 - j]);
        A[31 - j] = reverse32Bit(tmp);
    }
}

void Gemm::transposeBinary(uint32_t * const &A, uint32_t * const &B, const int &n, const int &m, const int &lda, const int &ldb, const int &blockSize)
{
    (void)blockSize;
#pragma omp parallel for num_threads(OMP_THREAD)
    for (int i = 0; i < n; i+=32)
    {
        int j;
        for ( j = 0; j < m; j+=32)
        {
            int aIndex  = i*lda + j;
            int bIndex  = j*ldb + i;

            Gemm::transpose32x32ReversedDiag(&A[aIndex/32], &B[bIndex/32], lda/32, ldb/32);
        }

        for (; j < m; ++j)
        {
            if(Gemm::getBit(reinterpret_cast<const uint8_t* const>(A), i*lda+j))
            {
                Gemm::setBit(reinterpret_cast<uint8_t* const>(B),j*ldb+i);
            }
        }
    }
}

int Gemm::binTransposeAlinInput(int k, int n, float *b, char **tBitInput, int ldbAlign, int bitAlign)
{
    int newLdb        =  k + (ldbAlign - k%ldbAlign);
    int tInputSize    =  newLdb * bitAlign;
    memset(*tBitInput, 0, static_cast<size_t>(tInputSize)*sizeof(uint8_t));
    transposeBinary(reinterpret_cast<uint32_t*>(b), reinterpret_cast<uint32_t*>(*tBitInput), k,n,bitAlign, newLdb,8);
    return tInputSize;
}

void Gemm::transpose32x32ReversedDiag(uint32_t *const &A, uint32_t *const &B, const int &m, const int &n)
{
    unsigned ATmp[32];

    for (int i = 0; i < 32; ++i)
    {
        ATmp[i] = A[i * m];
    }
    transpose32Optimized(ATmp);

    for (int i = 0; i < 32; ++i)
    {
        B[i*n] = ATmp[i];
    }
}

void Gemm::transposeUint32(uint32_t * const &input, uint32_t * const &output, const int &inH,
                           const int &inW, const int &inAlign, const int &outAlign)
{
    for (int i = 0; i < inH; ++i)     
    {
        for (int j = 0; j < inW; ++j)   
        {
            output[j*outAlign/32 + i] = input[i*inAlign + j];
        }
    }
}

#ifdef USE_X86

void Gemm::gemmNNBinMeanTrans(int M, int N, int K, float ALPHA_UNUSED, unsigned char *A, int lda, unsigned char *B, int ldb, float *C, int ldc, float *meanArr)
{

    (void) ALPHA_UNUSED;
#pragma omp parallel for num_threads(OMP_THREAD)
    for (int i = 0; i < (M/2)*2; i += 2)
    {

        float meanVal0 = meanArr[i + 0];
        float meanVal1 = meanArr[i + 1];

        for (int j = 0; j < (N/2)*2; j += 2)
        {

            const int bitStep = 256;
            __m256i countSum_0 = _mm256_set1_epi8(0);
            __m256i countSum_1 = _mm256_set1_epi8(0);
            __m256i countSum_2 = _mm256_set1_epi8(0);
            __m256i countSum_3 = _mm256_set1_epi8(0);

            for (int k = 0; k < K; k += bitStep) {   

                __m256i aBit256_0 = _mm256_loadu_si256(reinterpret_cast<__m256i *>((A + ((i + 0)*lda + k) / 8)));
                __m256i bBit256_0 = _mm256_loadu_si256(reinterpret_cast<__m256i *>((B + ((j + 0)*ldb + k) / 8)));

                __m256i aBit256_1 = _mm256_loadu_si256(reinterpret_cast<__m256i *>((A + ((i + 1)*lda + k) / 8)));
                __m256i bBit256_1 = _mm256_loadu_si256(reinterpret_cast<__m256i *>((B + ((j + 1)*ldb + k) / 8)));

                xnorAvx2Popcnt(aBit256_0, bBit256_0, &countSum_0);
                xnorAvx2Popcnt(aBit256_0, bBit256_1, &countSum_1);

                xnorAvx2Popcnt(aBit256_1, bBit256_0, &countSum_2);
                xnorAvx2Popcnt(aBit256_1, bBit256_1, &countSum_3);
            }

            int count0 = getCountMula(countSum_0);
            int count1 = getCountMula(countSum_1);
            int count2 = getCountMula(countSum_2);
            int count3 = getCountMula(countSum_3);

            const int f1 = (K % bitStep == 0) ? 0 : (bitStep - (K % bitStep));
            count0 = count0 - f1;    
            count1 = count1 - f1;
            count2 = count2 - f1;
            count3 = count3 - f1;
            C[i*ldc + (j + 0)] = (2 * count0 - K) * meanVal0;
            C[i*ldc + (j + 1)] = (2 * count1 - K) * meanVal0;
            C[(i + 1)*ldc + (j + 0)] = (2 * count2 - K) * meanVal1;
            C[(i + 1)*ldc + (j + 1)] = (2 * count3 - K) * meanVal1;
        }

        for (int iD = 0; iD < 2; ++iD)
        {
            float meanVal = meanArr[i + iD];
            for (int j = (N / 2) * 2; j < N; j += 1)
            {

                const int bit_step = 256;
                __m256i count_sum = _mm256_set1_epi8(0);

                for (int k = 0; k < K; k += bit_step)
                {

                    __m256i a_bit256_0 = _mm256_loadu_si256(reinterpret_cast<__m256i *>((A + ((i + iD + 0)*lda + k) / 8)));
                    __m256i b_bit256_0 = _mm256_loadu_si256(reinterpret_cast<__m256i *>((B + ((j + 0)*ldb + k) / 8)));
                    xnorAvx2Popcnt(a_bit256_0, b_bit256_0, &count_sum);
                }
                int count = getCountMula(count_sum);
                const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
                count = count - f1;    
                C[(i + iD)*ldc + j] = (2 * count - K) * meanVal;
            }
        }
    }

    for (int i = (M / 2) * 2; i < M; i += 1)
    {
        float mean_val = meanArr[i];
        for (int j = 0; j < N; j += 1)
        {

            const int bit_step = 256;
            __m256i count_sum = _mm256_set1_epi8(0);

            for (int k = 0; k < K; k += bit_step)
            {

                __m256i a_bit256_0 = _mm256_loadu_si256(reinterpret_cast<__m256i *>((A + ((i + 0)*lda + k) / 8)));
                __m256i b_bit256_0 = _mm256_loadu_si256(reinterpret_cast<__m256i *>((B + ((j + 0)*ldb + k) / 8)));
                xnorAvx2Popcnt(a_bit256_0, b_bit256_0, &count_sum);
            }
            int count = getCountMula(count_sum);
            const int f1 = (K % bit_step == 0) ? 0 : (bit_step - (K % bit_step));
            count = count - f1;    
            C[i*ldc + j] = (2 * count - K) * mean_val;
        }
    }
}
#endif
}
