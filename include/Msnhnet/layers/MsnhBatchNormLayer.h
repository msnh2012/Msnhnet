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
    BatchNormLayer(const int &batch, const int &width, const int &height, const int &channel, const ActivationType &activation, const std::vector<float> &actParams);

   float       *scales             =   nullptr;
    float       *biases             =   nullptr;
    float       *rollMean           =   nullptr;
    float       *rollVariance       =   nullptr;
    float       *activationInput    =   nullptr;

   int         nBiases             =   0;
    int         nScales             =   0;
    int         nRollMean           =   0;
    int         nRollVariance       =   0;

   virtual void forward(NetworkState &netState);

   static void addBias(float *const &output, float *const &biases, const int &batch, const int &channel, const int &whSize);
    static void scaleBias(float *const &output, float *const &scales, const int &batch, const int &channel, const int &whSize);

   void resize(int width, int height);

   void loadAllWeigths(std::vector<float> &weights);

   void loadScales(float *const &weights, const int& len);
    void loadBias(float *const &bias, const int& len);
    void loadRollMean(float *const &rollMean, const int& len);
    void loadRollVariance(float *const &rollVariance, const int& len);
    ~BatchNormLayer();
};
}

#endif 

