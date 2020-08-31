#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{

class ConvolutionLayerSgemm
{
public:
    //bottom: src, inWidth, inHeight, inChannel
    //top: dest, outWidth, outHeight, outChannel
    void convolution_im2col_sgemm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel,
                            const int &kernelW, const int &kernelH,int float* &dest, const int &outWidth, const int &outHeight, const int &outChannel);
};

}
