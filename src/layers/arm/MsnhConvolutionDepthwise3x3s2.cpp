#define USE_ARM 1
#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolutionDepthwise3x3s2.h"
namespace Msnhnet
{
    void ConvolutionalDepthwiseLayerArm3x3s2::convdepthwise3x3s2Neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                                 float* &dest, const int &outWidth, const int &outHeight, const int &outChannel){
        int ccOutChannel = outChannel >> 1;
        int ccRemainOutChannel = ccOutChannel << 1;

        const int in_size = inWidth * inHeight;
        const int out_size = outWidth * outHeight;
        const int group = inChannel;
#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif 
        for(int g = 0; g < group; g++){
            float *dest0 = dest + g * out_size;
            const float* kernel0 = kernel + g * 9;

            float* destptr0 = dest0;
            float* destptr0_next = destptr0 + outWidth;

            const float* src0 = src + g * in_size;

            const float* r0 = src0;
            const float* r1 = src0 + inWidth;
            const float* r2 = src0 + inWidth * 2;
            const float* r3 = src0 + inWidth * 3;

#if USE_NEON
            float32x4_t k012 = vld1q_f32(k0);
            float32x4_t k345 = vld1q_f32(k0 + 3);
            float32x4_t k678 = vld1q_f32(k0 + 6);
            k012 = vsetq_lane_f32(0.f, k012, 3);
            k345 = vsetq_lane_f32(0.f, k345, 3);
            k678 = vsetq_lane_f32(0.f, k678, 3);
#else
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;
#endif 

            int i = 0;
            
            for(; i < outHeight; i++){
#if USE_NEON
                int nn = outWidth >> 2;
                int remain = outWidth & 3;
#else
                int remain = outWidth;
#endif

#if USE_NEON

#if __aarch64__
                throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
#else  
                if(nn > 0){
                    asm volatile(

                    );
                }
#endif     
#endif
                for(; remain > 0; remain--){
                    
                }

            }
        }
    }
}

#endif