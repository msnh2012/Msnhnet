#ifdef USE_ARM
#include "Msnhnet/layers/arm/aarch64/MsnhActivation.h"
namespace Msnhnet
{

void ActivationLayerArm::logisticActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::loggyActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){

}

void ActivationLayerArm::reluActivateV8(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){
    int in_size = inWidth * inHeight;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int i = 0; i < inChannel; i++){

        float *srcPtr = src + i * in_size;

        #if USE_ARM
                int nn = in_size >> 2;
                int remain = in_size - (nn << 2);
        #else
                int remain = in_size;
        #endif

        #if USE_ARM
            if(nn > 0){
               
                asm volatile(
                    "movi       v0.2d, #0           \n"
                    "0:                             \n"
                    "prfm       pldl1keep, [%0, #128]   \n"
                    "ld1        {v1.4s}, [%0]       \n"
                    "fmax       v1.4s, v1.4s, v0.4s \n"
                    "st1        {v1.4s}, [%0], #16       \n"
                    "subs       %w1, %w1, #1        \n"
                    "bne        0b                  \n"
                    
                    : "=r"(srcPtr), // %0
                    "=r"(nn)        // %1
                    : "0"(srcPtr),
                    "1"(nn)
                    : "cc", "memory", "v0", "v1"
                );
            }
        #endif

        for(; remain > 0; remain--){
            *srcPtr = std::max(*srcPtr, 0.f);
            srcPtr++;
        }
    }
}

void ActivationLayerArm::relu6ActivateV8(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel){
    int in_size = inWidth * inHeight;

    const float six = 6.0;

#if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int i = 0; i < inChannel; i++){

        float *srcPtr = src + i * in_size;

        #if USE_ARM
                int nn = in_size >> 2;
                int remain = in_size - (nn << 2);
        #else
                int remain = in_size;
        #endif

        #if USE_ARM
            if(nn > 0){
                    asm volatile(
                        "movi       v0.2d, #0           \n"
                        "dup        v1.4s, %w4          \n"
                        "0:                             \n"
                        "prfm       pldl1keep, [%0, #128]   \n"
                        "ld1        {v2.4s}, [%0]       \n"
                        "fmax       v2.4s, v2.4s, v0.4s \n"
                        "fmin       v2.4s, v2.4s, v1.4s \n"
                        "st1        {v2.4s}, [%0], #16       \n"
                        "subs       %w1, %w1, #1        \n"
                        "bne        0b                  \n"

                        : "=r"(srcPtr),     // %0
                        "=r"(nn)            // %1
                        : "0"(srcPtr),
                        "1"(nn),
                        "r"(six)            // %w4
                        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4"
                    );
            }
        #endif

        for(; remain > 0; remain--){
            *srcPtr = std::min(std::max(*srcPtr, 0.f), six);
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

void ActivationLayerArm::leakyActivateV8(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel, const float &params){
    int in_size = inWidth * inHeight;

    #if USE_OMP
    #pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int i = 0; i < inChannel; i++){

        float *srcPtr = src + i * in_size;

        #if USE_ARM
                int nn = in_size >> 2;
                int remain = in_size - (nn << 2);
        #else
                int remain = in_size;
        #endif

        #if USE_ARM
            if(nn > 0){
                
                asm volatile(
                    "movi       v0.2d, #0           \n"
                    "dup        v1.4s, %w4          \n"
                    "0:                             \n"
                    "prfm       pldl1keep, [%0, #128]   \n"
                    "ld1        {v2.4s}, [%0]       \n"

                    "fcmge      v3.4s, v0.4s, v2.4s     \n"
                    "fmul       v4.4s, v1.4s, v2.4s     \n"
                    "bit        v2.16b, v4.16b, v3.16b  \n"
                    "st1        {v2.4s}, [%0], #16  \n"

                    "subs       %w1, %w1, #1        \n"
                    "bne        0b                  \n"
                    : "=r"(srcPtr), // %0
                    "=r"(nn) // %1
                    : "0"(srcPtr),
                    "1"(nn),
                    "r"(params) // %w4
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4"
                );
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
            break;
        case LOGGY:
            ActivationLayerArm::loggyActivate(src, inWidth, inHeight, inChannel); 
            break;
        case RELU:
            ActivationLayerArm::reluActivate(src, inWidth, inHeight, inChannel); 
            break;
        case RELU6:
            ActivationLayerArm::relu6Activate(src, inWidth, inHeight, inChannel); 
            break;
        case ELU:
            ActivationLayerArm::eluActivate(src, inWidth, inHeight, inChannel); 
            break;
        case SELU:
            ActivationLayerArm::seluActivate(src, inWidth, inHeight, inChannel); 
            break;
        case RELIE:
            ActivationLayerArm::relieActivate(src, inWidth, inHeight, inChannel);
            break;
        case RAMP:
            ActivationLayerArm::rampActivate(src, inWidth, inHeight, inChannel); 
            break;
        case LEAKY:
            ActivationLayerArm::leakyActivate(src, inWidth, inHeight, inChannel, params); 
            break;
        case TANH:
            ActivationLayerArm::tanhActivate(src, inWidth, inHeight, inChannel); 
            break;
        case PLSE:
            ActivationLayerArm::plseActivate(src, inWidth, inHeight, inChannel); 
            break;
        case STAIR:
            ActivationLayerArm::stairActivate(src, inWidth, inHeight, inChannel); 
            break;
        case HARDTAN:
            ActivationLayerArm::hardtanActivate(src, inWidth, inHeight, inChannel); 
            break;
        case LHTAN:
            ActivationLayerArm::lhtanActivate(src, inWidth, inHeight, inChannel); 
            break;
        case SOFT_PLUS:
            ActivationLayerArm::softplusActivate(src, inWidth, inHeight, inChannel, params); 
            break;
        case MISH:
            ActivationLayerArm::mishActivate(src, inWidth, inHeight, inChannel); 
            break;
        case SWISH:
            ActivationLayerArm::swishActivate(src, inWidth, inHeight, inChannel); 
            break;
        case NONE:
            break ;
        default:
            throw Exception(1, "Error: Activation Function!", __FILE__, __LINE__, __FUNCTION__);
    }

}

}
#endif
