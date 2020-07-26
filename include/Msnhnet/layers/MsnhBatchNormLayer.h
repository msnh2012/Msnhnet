#ifndef MSNHBATCHNORMLAYER_H
#define MSNHBATCHNORMLAYER_H

#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/layers/MsnhActivations.h"
#include <stdlib.h>
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API BatchNormLayer : public BaseLayer
{
public:
    BatchNormLayer(const int &_batch, const int &_width, const int &_height, const int &_channel, const ActivationType &_activation, const std::vector<float> &_actParams);

    virtual void forward(NetworkState &netState);

    static void addBias(float *const &_output, float *const &_biases, const int &_batch, const int &_channel, const int &whSize);
    static void scaleBias(float *const &_output, float *const &_scales, const int &_batch, const int &_channel, const int &whSize);

    void resize(int _width, int _height);

    void loadAllWeigths(std::vector<float> &weights);

    void loadScales(float *const &weights, const int& len);
    void loadBias(float *const &bias, const int& len);
    void loadRollMean(float *const &_rollMean, const int& len);
    void loadRollVariance(float *const &_rollVariance, const int& len);
    ~BatchNormLayer();

    float *getScales() const;

    float *getBiases() const;

    float *getRollMean() const;

    float *getRollVariance() const;

    float *getActivationInput() const;

    int getNBiases() const;

    int getNScales() const;

    int getNRollMean() const;

    int getNRollVariance() const;

protected:
    float       *_scales             =   nullptr;
    float       *_biases             =   nullptr;
    float       *_rollMean           =   nullptr;
    float       *_rollVariance       =   nullptr;
    float       *_activationInput    =   nullptr;

    int         _nBiases             =   0;
    int         _nScales             =   0;
    int         _nRollMean           =   0;
    int         _nRollVariance       =   0;
};
}

#endif 

