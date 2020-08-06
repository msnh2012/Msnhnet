#ifndef MSNHCONVOLUTIONALLAYER_H
#define MSNHCONVOLUTIONALLAYER_H
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/core/MsnhGemm.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/layers/MsnhActivations.h"
#include "Msnhnet/layers/MsnhBatchNormLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

#ifdef USE_NNPACK
#include <nnpack.h>
#endif

namespace Msnhnet
{
class MsnhNet_API ConvolutionalLayer:public BaseLayer
{
public:

    ConvolutionalLayer(const int &_batch, const int &_steps, const int &_height, const int &_width, const int &_channel, const int &_num, const int &_groups,
                      const int &_kSizeX, const int &_kSizeY, const int &_strideX, const int &_strideY, const int &_dilationX, const int &_dilationY, const int &_paddingX, const int &_paddingY, ActivationType _activation, const std::vector<float> &_actParams,
                      const int &_batchNorm,  const int &_useBias, const int &_binary, const int &_xnor, const int &_useBinOutput, const int &_groupIndex,
                      const int &_antialiasing, ConvolutionalLayer *const &_shareLayer, const int &assistedExcitation, const int &deform);
    ~ConvolutionalLayer();

    int convOutHeight();
    int convOutWidth();

    int getConvWorkSpaceSize();
    int getWorkSpaceSize32();
    int getWorkSpaceSize16();

    static void  addBias(float *const &_output, float *const &_biases, const int &_batch, const int &_num, const int &whSize);
    static void  scaleBias(float *const &_output, float *const &_scales, const int &_batch, const int &_num, const int &whSize);

    void binarizeWeights(float *const &_weights, const int &_num, const int &wtSize, float *const &_binary);
    void cpuBinarize(float *const &x, const int &xNum, float *const &_binary);
    void swapBinary();

    void forward(NetworkState &netState);
#ifdef USE_GPU
    void forwardGPU(NetworkState &netState);
#endif
    void loadAllWeigths(std::vector<float> &_weights);

    void loadScales(float *const &_weights, const int& len);
    void loadBias(float *const &bias, const int& len);
    void loadWeights(float *const &_weights, const int& len);
    void loadRollMean(float * const &_rollMean, const int &len);
    void loadRollVariance(float * const &_rollVariance, const int &len);

    float *getWeights() const;

    float *getBiases() const;

    float *getScales() const;

    float *getRollMean() const;

    float *getRollVariance() const;

    char *getCWeights() const;

    float *getBinaryInputs() const;

    float *getBinaryWeights() const;

    float *getMeanArr() const;

    uint32_t *getBinRePackedIn() const;

    char *getTBitInput() const;

    char *getAlignBitWeights() const;

    int getBitAlign() const;

    int getLdaAlign() const;

    int getUseBias() const;

    int getNScales() const;

    int getNRollMean() const;

    int getNRollVariance() const;

    int getNBiases() const;

    int getNWeights() const;

    int getGroups() const;

    int getGroupIndex() const;

    int getXnor() const;

    int getBinary() const;

    int getUseBinOutput() const;

    int getSteps() const;

    int getAntialiasing() const;

    int getAssistedExcite() const;

    int getKSizeX() const;

    int getKSizeY() const;

    int getStride() const;

    int getStrideX() const;

    int getStrideY() const;

    int getPaddingX() const;

    int getPaddingY() const;

    int getDilationX() const;

    int getDilationY() const;

    int getBatchNorm() const;

protected:
    float       *_weights            =   nullptr;
    float       *_biases             =   nullptr;
    ConvolutionalLayer* _shareLayer  =   nullptr;

    float       *_scales             =   nullptr;
    float       *_rollMean           =   nullptr;
    float       *_rollVariance       =   nullptr;

    char        *_cWeights           =   nullptr;
    float       *_binaryInputs       =   nullptr;
    float       *_binaryWeights      =   nullptr;
    float       *_meanArr            =   nullptr;
    uint32_t    *_binRePackedIn      =   nullptr;
    char        *_tBitInput          =   nullptr;
    char        *_alignBitWeights    =   nullptr;

#ifdef USE_GPU
    float       *_gpuWeights         =   nullptr;
    float       *_gpuBiases          =   nullptr;

    float       *_gpuScales          =   nullptr;
    float       *_gpuRollMean        =   nullptr;
    float       *_gpuRollVariance    =   nullptr;

#endif

    int         _bitAlign            =   0;
    int         _ldaAlign            =   0;

    int         _useBias             =   1;

    int         _nScales             =   0;
    int         _nRollMean           =   0;
    int         _nRollVariance       =   0;

    int         _nBiases             =   0;
    int         _nWeights            =   0;
    int         _groups              =   0;
    int         _groupIndex          =   0;

    int         _xnor                =   0;
    int         _binary              =   0;
    int         _useBinOutput        =   0;
    int         _steps               =   0;

    int         _antialiasing        =   0;
    int         _assistedExcite      =   0;

    int         _kSizeX              =   0;
    int         _kSizeY              =   0;
    int         _stride              =   0;
    int         _strideX             =   0;
    int         _strideY             =   0;
    int         _paddingX            =   0;
    int         _paddingY            =   0;
    int         _dilationX           =   0;
    int         _dilationY           =   0;
    int         _batchNorm           =   0;
};
}

#endif 

