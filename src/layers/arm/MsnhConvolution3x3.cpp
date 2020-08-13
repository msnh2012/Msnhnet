#include "Msnhnet/layers/arm/MsnhConvolution3x3.h"
#if ARM_NEON
#include <arm_neon.h>
#endif

namespace Msnhnet
{

void ConvolutionalLayerArm3x3::conv3x3s1_neon(float *src, int inw, int inh,  int inch,  float *kernel, int kw, int kh, float *dest, int outw, int outh, int outch){
    int cc_outch = outch >> 1;
    int cc_remain_outch = outch << 1;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif 
    for(int cc = 0; cc < cc_outch; cc++){
        int c = cc * 2;
        float *out0 = dest + cc * (outw * outh);
        float *out1 =  dest + (cc + 1) * (outw * outh);
        float bias0 = 0;
        float bias1 = 0;
        float *k0 = kernel + cc * inch * 3 * 3;
        float *k1 = kernel + (cc + 1) * inch * 3 * 3;

        for(int q = 0; q < inch; q++){
            
        }
    }
}

}