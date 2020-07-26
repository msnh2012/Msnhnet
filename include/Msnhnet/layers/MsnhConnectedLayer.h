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
    ConnectedLayer(const int &_batch, const int &steps, const int &_inputNum, const int &_outputNum,
                   const ActivationType &_activation, const std::vector<float> &_actParams, const int &_batchNorm);

    ~ConnectedLayer();

    virtual void forward(NetworkState &netState);

    void loadAllWeigths(std::vector<float> &_weights);

    void loadScales(float *const &_weights, const int& len);
    void loadBias(float *const &bias, const int& len);
    void loadWeights(float *const &_weights, const int& len);
    void loadRollMean(float *const &_rollMean, const int& len);
    void loadRollVariance(float *const &_rollVariance, const int& len);

    float *getWeights() const;

    float *getBiases() const;

    float *getScales() const;

    float *getRollMean() const;

    float *getRollVariance() const;

    int getNBiases() const;

    int getNWeights() const;

    int getNScales() const;

    int getNRollMean() const;

    int getNRollVariance() const;

    int getKSize() const;

    int getStride() const;

    int getStrideX() const;

    int getStrideY() const;

    int getPadding() const;

    int getDilation() const;

    int getBatchNorm() const;

protected:
    float       *_weights            =   nullptr;
    float       *_biases             =   nullptr;
    float       *_scales             =   nullptr;
    float       *_rollMean           =   nullptr;
    float       *_rollVariance       =   nullptr;

    int         _nBiases             =   0;
    int         _nWeights            =   0;
    int         _nScales             =   0;
    int         _nRollMean           =   0;
    int         _nRollVariance       =   0;

    int         _kSize               =   0;
    int         _stride              =   0;
    int         _strideX             =   0;
    int         _strideY             =   0;
    int         _padding             =   0;
    int         _dilation            =   0;
    int         _batchNorm           =   0;
};
}

#endif 

