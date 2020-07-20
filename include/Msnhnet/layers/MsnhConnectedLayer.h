#ifndef MSNHCONNECTEDLAYER_H
#define MSNHCONNECTEDLAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhActivations.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/core/MsnhGemm.h"
#include "Msnhnet/layers/MsnhConvolutionalLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{

class MsnhNet_API ConnectedLayer : public BaseLayer
{
public:
    ConnectedLayer(const int &batch, const int &steps, const int &inputNum, const int &outputNum,
                   const ActivationType &activation, const std::vector<float> &actParams, const int &batchNorm);

    ~ConnectedLayer();

    float       *weights            =   nullptr;
    float       *biases             =   nullptr;
    float       *scales             =   nullptr;
    float       *rollMean           =   nullptr;
    float       *rollVariance       =   nullptr;

    int         nBiases             =   0;
    int         nWeights            =   0;
    int         nScales             =   0;
    int         nRollMean           =   0;
    int         nRollVariance       =   0;

    int         kSize               =   0;
    int         stride              =   0;
    int         strideX             =   0;
    int         strideY             =   0;
    int         padding             =   0;
    int         dilation            =   0;
    int         batchNorm           =   0;

    virtual void forward(NetworkState &netState);

    void loadAllWeigths(std::vector<float> &weights);

    void loadScales(float *const &weights, const int& len);
    void loadBias(float *const &bias, const int& len);
    void loadWeights(float *const &weights, const int& len);
    void loadRollMean(float *const &rollMean, const int& len);
    void loadRollVariance(float *const &rollVariance, const int& len);
};
}

#endif 
