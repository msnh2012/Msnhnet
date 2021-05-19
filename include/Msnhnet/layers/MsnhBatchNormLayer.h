#ifndef MSNHBATCHNORMLAYER_H
#define MSNHBATCHNORMLAYER_H

#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/net/MsnhNetwork.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/layers/MsnhActivations.h"
#include <stdlib.h>
#include "Msnhnet/utils/MsnhExport.h"

#ifdef USE_ARM
#include "Msnhnet/layers/arm/armv7a/MsnhBatchNorm.h"
#endif

#ifdef USE_OPENCL
#include "Msnhnet/layers/opencl/MsnhBatchNormCL.h"
#endif

namespace Msnhnet
{
class MsnhNet_API BatchNormLayer : public BaseLayer
{
public:
    BatchNormLayer(const int &batch, const int &width, const int &height, const int &channel, const ActivationType &activation, const float &eps, const std::vector<float> &actParams);

    virtual void forward(NetworkState &netState);
    virtual void mallocMemory();

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

#ifdef USE_OPENCL
    virtual void forwardCL(NetworkState &netState);
#endif

    static void addBias(float *const &output, float *const &biases, const int &batch, const int &channel, const int &whSize);
    static void scaleBias(float *const &output, float *const &scales, const int &batch, const int &channel, const int &whSize);

    void resize(int _width, int _height);

    void loadAllWeigths(std::vector<float> &weights);

    void saveAllWeights(const int &mainIdx, const int &branchIdx=-1, const int &branchIdx1=-1);

    void loadScales(float *const &weights, const int& len);
    void loadBias(float *const &bias, const int& len);
    void loadRollMean(float *const &rollMean, const int& len);
    void loadRollVariance(float *const &rollVariance, const int& len);
    void loadPreluWeights(float *const &weights, const int& len); 

    ~BatchNormLayer();

    float *getScales() const;
    float *getBiases() const;
    float *getRollMean() const;
    float *getRollVariance() const;
    float *getActivationInput() const;
    float *getPreluWeights() const;

    int getNBiases() const;
    int getNScales() const;
    int getNRollMean() const;
    int getNRollVariance() const;
    int getNPreluWeights() const;

protected:
    float       *_scales             =   nullptr;
    float       *_biases             =   nullptr;
    float       *_rollMean           =   nullptr;
    float       *_rollVariance       =   nullptr;
    float       *_activationInput    =   nullptr;
    float       *_preluWeights       =   nullptr;

#ifdef USE_GPU
    float       *_gpuScales          =   nullptr;
    float       *_gpuBiases          =   nullptr;
    float       *_gpuRollMean        =   nullptr;
    float       *_gpuRollVariance    =   nullptr;
    float       *_gpuPreluWeights    =   nullptr;
#endif

#ifdef USE_OPENCL
    cl_kernel   _kernel;
    cl_kernel   _kernel_act;
    
    cl_int      status;
    
    cl_mem      _clBiases;
    cl_mem      _clScales;
    cl_mem      _clRollMean;
    cl_mem      _clRollVariance;
#endif 

    int         _nPreluWeights       =   0;
    int         _nBiases             =   0;
    int         _nScales             =   0;
    int         _nRollMean           =   0;
    int         _nRollVariance       =   0;
    float       _eps                 =   0.00001f;
};
}

#endif 

