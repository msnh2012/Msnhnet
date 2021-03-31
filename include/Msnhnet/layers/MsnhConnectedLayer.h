#ifndef MSNHCONNECTEDLAYER_H
#define MSNHCONNECTEDLAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/layers/MsnhActivations.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/core/MsnhGemm.h"
#include "Msnhnet/layers/MsnhConvolutionalLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

#ifdef USE_OPENCL
#include "Msnhnet/layers/opencl/MsnhConnectedCL.h"
#endif 

namespace Msnhnet
{

class MsnhNet_API ConnectedLayer : public BaseLayer
{
public:
    ConnectedLayer(const int &batch, const int &steps, const int &inputNum, const int &outputNum,
                   const ActivationType &activation, const std::vector<float> &actParams, const int &batchNorm, const float &bnEps, const int &useBias);

    ~ConnectedLayer();

    virtual void forward(NetworkState &netState);

    virtual void mallocMemory();

    virtual void saveAllWeights(const int &mainIdx, const int &branchIdx=-1, const int &branchIdx1=-1);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

#ifdef USE_OPENCL
    void forwardCL(NetworkState &netState);
#endif

    void loadAllWeigths(std::vector<float> &weights);

    void loadScales(float *const &weights, const int& len);

    void loadBias(float *const &bias, const int& len);

    void loadWeights(float *const &weights, const int& len);

    void loadRollMean(float *const &rollMean, const int& len);

    void loadRollVariance(float *const &rollVariance, const int& len);

    void loadPreluWeights(float *const &weights, const int& len); 

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
    float       *_preluWeights       =   nullptr;

    int         _useBias             =   0;

#ifdef USE_GPU
    float       *_gpuWeights         =   nullptr;
    float       *_gpuBiases          =   nullptr;

    float       *_gpuScales          =   nullptr;
    float       *_gpuRollMean        =   nullptr;
    float       *_gpuRollVariance    =   nullptr;
    float       *_gpuPreluWeights    =   nullptr;
#endif

#ifdef USE_OPENCL

    cl_kernel   _kernel_con;
    cl_kernel   _kernel_act;
    cl_kernel   _kernel_bn;

    cl_int      status;

    cl_mem      _clWeights;
    cl_mem      _clBiases;
    cl_mem      _clScales;
    cl_mem      _clRollMean;
    cl_mem      _clRollVariance;
    cl_mem      _clPreluWeights;

#endif

    int         _nBiases             =   0;
    int         _nWeights            =   0;
    int         _nScales             =   0;
    int         _nRollMean           =   0;
    int         _nRollVariance       =   0;
    int         _nPreluWeights       =   0;

    int         _kSize               =   0;
    int         _stride              =   0;
    int         _strideX             =   0;
    int         _strideY             =   0;
    int         _padding             =   0;
    int         _dilation            =   0;
    int         _batchNorm           =   0;

    int         _totalBatch          =   0;
    float       _bnEps               =   0.00001f;

};
}

#endif 

