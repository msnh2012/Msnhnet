#ifndef MSNHGEMM_H
#define MSNHGEMM_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhSimd.h"
#include "Msnhnet/utils/MsnhExport.h"
#ifdef USE_GPU
#include "Msnhnet/core/cuda/MsnhGemmGPU.h"
#endif

namespace Msnhnet
{
class MsnhNet_API Gemm
{
public:

    static void repackInput(float *const &input, float *const &rePackedInput,
                            const int &width, const int &height, const int &channel);

    static void float2Bit(float *const &input, uint8_t *const &output, size_t size);

    static void cpuIm2col(float *const &input, const int &channelNum, const int &height, const int &width,
                          const int &kSize,const int &stride, const int &padding, float *const &output);

    static void cpuCol2Im(float *const &input, const int &channelNum, const int &height, const int &width,
                           const int &kSizeX, const int &kSizeY, const int &strideX, const int &strideY,
                           const int &paddingX, const int &paddingY, float *const &output);

    inline static int is_a_ge_zero_and_a_lt_b(const int &a, const int &b)
    {
        return static_cast<unsigned>(a) < static_cast<unsigned>(b);
    }

    static void cpuIm2colEx(float *input, const int &channelNum, const int &height, const int &width,
                            const int &kernelH, const int &kernelW, const int &padH, const int &padW,
                            const int &strideH,  const int &strideW, const int &dilationH, const int &dilationW,
                            float *output);

    static void cpuCol2ImEx(float *input, const int &channelNum, const int &height, const int &width,
                            const int &kernelH, const int &kernelW, const int &padH, const int &padW,
                            const int &strideH,  const int &strideW, const int &dilationH, const int &dilationW,
                            float *output);

    static void cpuIm2colWithAvx(float * const &input, const int &channelNum, const int &height, const int &width,const int &kSize,
                                 const int &stride, const int &padding, float * const &output,const bool &supportAvxAndFma);

    static void cpuIm2colBinWithAvx(float * const &input, const int &channelNum, const int &height, const int &width,const int &kSize,
                                    const int &stride, const int &padding, float * const &output, const int &bitAlign, const bool &supportAvxAndFma);

    static float img2ColGetPixel(float *const &input, const int &height, const int &width, const int &row,
                                 const int &col,const int &channel, const int &padding);

    static inline void setBit(uint8_t *const &dst, const int &index)
    {
        int dstI  = index / 8;
        int dstShift = index % 8;
        dst[dstI]   |= 1 << dstShift;
    }

    static inline uint8_t getBit(const uint8_t *const &src, int index)
    {
        int srcI      = index / 8;
        int srcShift  = index % 8;
        uint8_t val   = (src[srcI] & (1 << srcShift)) > 0;
        return val;
    }

    static void cpuGemm(const int &TA,   const int &TB, const int &M, const int &N, const int &K, const float &ALPHA,
                        float *const &A, const int &lda,
                        float *const &B, const int &ldb,
                        const float &BETA,
                        float *const &C, const int &ldc,
                        const bool &supportAvxAndFma);

    static void cpuGemm(const int &TA,   const int &TB, const int &M, const int &N, const int &K, const double &ALPHA,
                        double *const &A, const int &lda,
                        double *const &B, const int &ldb,
                        const float &BETA,
                        double *const &C, const int &ldc,
                        const bool &supportAvxAndFma);

    static void cpuGemmNN(const int &M, const int &N, const int &K, const float &ALPHA,
                          float *const &A, const int &lda,
                          float *const &B, const int &ldb,
                          float *const &C, const int &ldc,
                          const bool &supportAvxAndFma);

    static void cpuGemmNN(const int &M, const int &N, const int &K, const double &ALPHA,
                          double *const &A, const int &lda,
                          double *const &B, const int &ldb,
                          double *const &C, const int &ldc,
                          const bool &supportAvxAndFma);

    static void cpuGemmTN(const int &M, const int &N, const int &K, const float &ALPHA,
                          float *const &A, const int &lda,
                          float *const &B, const int &ldb,
                          float *const &C, const int &ldc,
                          const bool &supportAvxAndFma);

    static void cpuGemmTN(const int &M, const int &N, const int &K, const double &ALPHA,
                          double *const &A, const int &lda,
                          double *const &B, const int &ldb,
                          double *const &C, const int &ldc,
                          const bool &supportAvxAndFma);

    static void cpuGemmNT(const int &M, const int &N, const int &K, const float &ALPHA,
                          float *const &A, const int &lda,
                          float *const &B, const int &ldb,
                          float *const &C, const int &ldc,
                          const bool &supportAvxAndFma);

    static void cpuGemmNT(const int &M, const int &N, const int &K, const double &ALPHA,
                          double *const &A, const int &lda,
                          double *const &B, const int &ldb,
                          double *const &C, const int &ldc,
                          const bool &supportAvxAndFma);

    static void cpuGemmTT(const int &M, const int &N, const int &K, const float &ALPHA,
                          float *const &A, const int &lda,
                          float *const &B, const int &ldb,
                          float *const &C, const int &ldc,
                          const bool &supportAvxAndFma);

    static void cpuGemmTT(const int &M, const int &N, const int &K, const double &ALPHA,
                          double *const &A, const int &lda,
                          double *const &B, const int &ldb,
                          double *const &C, const int &ldc,
                          const bool &supportAvxAndFma);

    static void cpuFastADotB(const int &n, float *const &A, float *const& B, float *const &C);

#define TILE_F32_M 4  

#define TILE_F32_N 16 

#define TILE_F32_K 16 

#define TILE_F64_M 4  

#define TILE_F64_N 8 

#define TILE_F64_K 16 

    static void cpuGemmNNFast(const int &M, const int &N, const int &K, const float &ALPHA,
                              float *const &A, const int &lda,
                              float *const &B, const int &ldb,
                              float *const &C, const int &ldc);

    static void cpuGemmNNFast(const int &M, const int &N, const int &K, const double &ALPHA,
                              double *const &A, const int &lda,
                              double *const &B, const int &ldb,
                              double *const &C, const int &ldc);

    static void cpuGemmTNFast(const int &M, const int &N, const int &K, const float &ALPHA,
                              float *const &A, const int &lda,
                              float *const &B, const int &ldb,
                              float *const &C, const int &ldc);

    static void cpuGemmTNFast(const int &M, const int &N, const int &K, const double &ALPHA,
                              double *const &A, const int &lda,
                              double *const &B, const int &ldb,
                              double *const &C, const int &ldc);

    static void swapVal(uint32_t &a0, uint32_t&a1, int &j, unsigned &m);

    static uint8_t lookup[16] ;

    static inline uint8_t reverse8Bit(const uint8_t &a)
    {
        return ((a * 0x0802U & 0x22110U) | (a * 0x8020U & 0x88440U)) * 0x10101U >> 16;
    }

    static inline uint8_t reverseByte1(const uint8_t &a)
    {
        return static_cast<uint8_t>(((a & 0x1) << 7) | ((a & 0x2) << 5) |
                                    ((a & 0x4) << 3) | ((a & 0x8) << 1) |
                                    ((a & 0x10) >> 1) | ((a & 0x20) >> 3) |
                                    ((a & 0x40) >> 5) | ((a & 0x80) >> 7));
    }

    static inline uint8_t reverseByte3(const uint8_t &n)
    {

        return static_cast<uint8_t>((lookup[n & 0b1111] << 4) | lookup[n >> 4]);
    }

    static inline uint32_t reverse32Bit(const uint32_t &a)
    {
        return  static_cast<uint32_t>((reverse8Bit(static_cast<uint8_t>(a >> 24)) << 0) |
                                      (reverse8Bit(static_cast<uint8_t>(a >> 16)) << 8) |
                                      (reverse8Bit(static_cast<uint8_t>(a >> 8)) << 16) |
                                      (reverse8Bit(static_cast<uint8_t>(a >> 0)) << 24));
    }

    static void transpose32Optimized(uint32_t *const& A, const int &num=32);

    static void transpose32x32ReversedDiag(uint32_t *const &A, uint32_t *const &B, const int &m, const int &n);

    static void transposeBinary(uint32_t *const &A, uint32_t *const &B, const int &n, const int &m,
                                const int &lda, const int &ldb, const int &blockSize);

    static int binTransposeAlinInput(int k, int n, float *b, char **tBitInput, int ldbAlign, int bitAlign);

    static void transposeUint32(uint32_t *const &input, uint32_t *const &output, const int &inH,
                                const int &inW, const int &inAlign, const int &outAlign);

#ifdef USE_X86

    static void gemmNNBinMeanTrans(int M, int N, int K, float ALPHA_UNUSED,
                                   unsigned char *A, int lda,
                                   unsigned char *B, int ldb,
                                   float *C, int ldc, float *mean_arr);

    static inline void xnorAvx2Popcnt(__m256i aBit256, __m256i bBit256, __m256i *countSum)
    {
        __m256i cBit256 = _mm256_set1_epi8(static_cast<char>(-1));
        __m256i xor256  = _mm256_xor_si256(aBit256, bBit256);  

        cBit256         = _mm256_andnot_si256(xor256, cBit256);       

        *countSum      = _mm256_add_epi64(count256(cBit256), *countSum);    

    }

    static inline int getCountMula(__m256i countSum)
    {

        return  static_cast<int>(_mm256_extract_epi64(countSum, 0)
                                 + _mm256_extract_epi64(countSum, 1)
                                 + _mm256_extract_epi64(countSum, 2)
                                 + _mm256_extract_epi64(countSum, 3));

    }

    static inline int popcnt128(__m128i n)
    {
        const __m128i nHi = _mm_unpackhi_epi64(n, n);
#if defined(_MSC_VER)
        return static_cast<int>(__popcnt64(_mm_cvtsi128_si64(n)) + __popcnt64(_mm_cvtsi128_si64(nHi)));
#elif defined(__APPLE__) && defined(__clang__)
        return _mm_popcnt_u64(_mm_cvtsi128_si64(n)) + _mm_popcnt_u64(_mm_cvtsi128_si64(nHi));
#else
        return __popcntq(_mm_cvtsi128_si64(n)) + __popcntq(_mm_cvtsi128_si64(nHi));
#endif
    }

    static inline int popcnt256(__m256i n)
    {
        return popcnt128(_mm256_extractf128_si256(n, 0)) + popcnt128(_mm256_extractf128_si256(n, 1));
    }

    static inline __m256i count256(__m256i v)
    {
        __m256i lookup   = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
                                            2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3,
                                            1, 2, 2, 3, 2, 3, 3, 4);

        __m256i low_mask = _mm256_set1_epi8(0x0f);

        __m256i lo = _mm256_and_si256(v, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
        __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
        __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
        __m256i total = _mm256_add_epi8(popcnt1, popcnt2);

        return _mm256_sad_epu8(total, _mm256_setzero_si256());
    }

    static inline int popcnt256_custom(__m256i n)
    {
        __m256i val = count256(n);

        return  static_cast<int>( _mm256_extract_epi64(val, 0)
                                  + _mm256_extract_epi64(val, 1)
                                  + _mm256_extract_epi64(val, 2)
                                  + _mm256_extract_epi64(val, 3));
    }
#endif

};

}

#endif 

