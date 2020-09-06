#include "MsnhNet/layers/arm/MsnhActivationLayerArm.h"
#include "iostrem"

namespace Msnhnet
{

void ActivationLayerArm::logisticActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::loggyActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::reluActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){
    int in_size = inWidth * inHeight;

    #if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int i = 0; i < inChannel; i++){

        float *srcPtr = src + i * in_size;

        #if USE_NEON
                int nn = in_size >> 2;
                int remain = in_size - nn << 2;
        #else
                int remain = in_size;
        #endif

        #if USE_NEON
            if(nn > 0){
                #if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
                #else
                    asm volatile(
                    "veor       q1, q0, q0          \n" //模拟0，more speed
                    
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q0, q0, q1          \n"
                    "vst1.f32   {d0-d1}, [%1 :128]! \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn), // %0
                    "=r"(srcPtr) // %1
                    : "0"(nn),
                    "1"(srcPtr)
                    : "cc", "memory", "q0", "q1");
                #endif
            }
        #endif

        for(; remain > 0; remain--){
            *srcPtr = std::max(*srcPtr, 0.f);
            srcPtr++;
        }
    }
}

void ActivationLayerArm::relu6Activate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){
    int in_size = inWidth * inHeight;

    const float six = 6.0;

    #if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int i = 0; i < inChannel; i++){

        float *srcPtr = src + i * in_size;

        #if USE_NEON
                int nn = in_size >> 2;
                int remain = in_size - nn << 2;
        #else
                int remain = in_size;
        #endif

        #if USE_NEON
            if(nn > 0){
                #if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
                #else
                    asm volatile(
                    "veor       q1, q0, q0          \n" //模拟0，more speed
                    "vdup.f32   q2, %4              \n"

                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"
                    "vmax.f32   q0, q0, q1          \n"
                    "vmin.f32   q0, q0, q2          \n"

                    "vst1.f32   {d0-d1}, [%1 :128]! \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn), // %0
                    "=r"(srcPtr) // %1
                    : "0"(nn),
                    "1"(srcPtr)
                    "r"(six) // %4
                    : "cc", "memory", "q0", "q1");
                #endif
            }
        #endif

        for(; remain > 0; remain--){
            *srcPtr = std::min(std::max(*srcPtr, 0.f), 6.0);
            srcPtr++;
        }
    }
}

void ActivationLayerArm::eluActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::seluActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::relieActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::rampActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::leakyActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel, const float &params){
    int in_size = inWidth * inHeight;

    #if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int i = 0; i < inChannel; i++){

        float *srcPtr = src + i * in_size;

        #if USE_NEON
                int nn = in_size >> 2;
                int remain = in_size - nn << 2;
        #else
                int remain = in_size;
        #endif

        #if USE_NEON
            if(nn > 0){
                #if __aarch64__
                    throw Exception(1, "Error: armv8 temporarily not supported!", __FILE__, __LINE__, __FUNCTION__);
                #else
                    asm volatile(
                    "veor       q1, q0, q0          \n"
                    // slope
                    "vdup.f32   q2, %4              \n"
                    
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]  \n"

                    // r[i] = a[i] <= b[i] ? 1: 0
                    "vcl e.f32   q3, q0, q1          \n"
                    // 
                    "vmul.f32   q4, q0, q2          \n"
                    // _p = vbslq_f32(_lemask, _ps, _p);
                    "vbit.32    q0, q4, q3          \n"

                    "vst1.f32   {d0-d1}, [%1 :128]! \n"

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn), // %0
                    "=r"(srcPtr) // %1
                    : "0"(nn),
                    "1"(srcPtr),
                    "r"(params) // %4
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4");
                #endif
            }
        #endif

        for(; remain > 0; remain--){
            if((*srcPtr) < 0){
                *srcPtr *= params;
            }
            srcPtr++;
        }
    }
}

void ActivationLayerArm::tanhActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::plseActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::stairActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::hardtanActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::lhtanActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::softplusActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel, const float &params){

}

void ActivationLayerArm::mishActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::swishActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::ActivationLayer(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel, const ActivationType &actType, const float &params){

    switch (actType){
        case LOGISTIC:
            ActivationLayerArm::logisticActivate(src, inWidth, inHeight, inChannel);
        case LOGGY:
            ActivationLayerArm::loggyActivate(src, inWidth, inHeight, inChannel);
        case RELU:
            ActivationLayerArm::reluActivate(src, inWidth, inHeight, inChannel);
        case RELU6:
            ActivationLayerArm::relu6Activate(src, inWidth, inHeight, inChannel);
        case ELU:
            ActivationLayerArm::eluActivate(src, inWidth, inHeight, inChannel);
        case SELU:
            ActivationLayerArm::seluActivate(src, inWidth, inHeight, inChannel);
        case RELIE:
            ActivationLayerArm::relieActivate(src, inWidth, inHeight, inChannel);
        case RAMP:
            ActivationLayerArm::rampActivate(src, inWidth, inHeight, inChannel);
        case LEAKY:
            ActivationLayerArm::leakyActivate(src, inWidth, inHeight, inChannel, params);
        case TANH:
            ActivationLayerArm::tanhActivate(src, inWidth, inHeight, inChannel);
        case PLSE:
            ActivationLayerArm::plseActivate(src, inWidth, inHeight, inChannel);
        case STAIR:
            ActivationLayerArm::stairActivate(src, inWidth, inHeight, inChannel);
        case HARDTAN:
            ActivationLayerArm::hardtanActivate(src, inWidth, inHeight, inChannel);
        case LHTAN:
            ActivationLayerArm::lhtanActivate(src, inWidth, inHeight, inChannel);
        case SOFT_PLUS:
            ActivationLayerArm::softplusActivate(src, inWidth, inHeight, inChannel, params);
        case MISH:
            ActivationLayerArm::mishActivate(src, inWidth, inHeight, inChannel);
        case SWISH:
            ActivationLayerArm::swishActivate(src, inWidth, inHeight, inChannel);
        case NONE:
            return ;
        default:
            return;
    }

}

}
