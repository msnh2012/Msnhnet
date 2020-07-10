#ifndef MSNHDECONVOLUTIONALLAYER_H
#define MSNHDECONVOLUTIONALLAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/core/MsnhGemm.h"
#include "Msnhnet/layers/MsnhConvolutionalLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API DeConvolutionalLayer:public BaseLayer
{
public:
    DeConvolutionalLayer(const int &batch, const int &height, const int &channel, const int &num,
                         const int &kSize, const int &stride, const ActivationType &activation, const std::vector<float> &actParams);
    float       *weights            =   nullptr;
    float       *biases             =   nullptr;
    float       *colImg             =   nullptr;
    int         kSize               =   0;
    int         stride              =   0;
    int         strideX             =   0;
    int         strideY             =   0;
    int         padding             =   0;
    int         dilation            =   0;
    int         batchNorm           =   0;
    virtual void forward(NetworkState &netState);

    int deConvOutHeight();
    int deConvOutWidth();
};
}
#endif 

