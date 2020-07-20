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
    DeConvolutionalLayer(const int &batch, const int &height, const int &width, const int &channel, const int &num,
                         const int &kSizeX, const int &kSizeY, const int &strideX, const int &strideY,const int &paddingX, const int &paddingY,
                         const ActivationType &activation, const std::vector<float> &actParams,const int &useBias);
    float       *weights            =   nullptr;
    float       *biases             =   nullptr;
    float       *colImg             =   nullptr;

    int         kSizeX              =   0;
    int         kSizeY              =   0;

    int         stride              =   0;
    int         strideX             =   0;
    int         strideY             =   0;

    int         paddingX            =   0;
    int         paddingY            =   0;

    int         useBias             =   1;

    int         nBiases             =   0;
    int         nWeights            =   0;

    virtual void forward(NetworkState &netState);
    virtual void loadAllWeigths(std::vector<float> &weights);
    void loadBias(float *const &bias, const int& len);
    void loadWeights(float *const &weights, const int& len);

    int deConvOutHeight();
    int deConvOutWidth();
};
}
#endif 
