#ifndef MSNHCONVOLUTIONALLAYER_H
#define MSNHCONVOLUTIONALLAYER_H
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/core/MsnhGemm.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/layers/MsnhActivations.h"
#include "Msnhnet/layers/MsnhBatchNormLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhConvolutionalLayerGPU.h"
#endif

#ifdef USE_ARM
#include "Msnhnet/layers/arm/MsnhConvolution3x3s1.h"
#include "Msnhnet/layers/arm/MsnhConvolution3x3s2.h"
#include "Msnhnet/layers/arm/MsnhConvolutionSgemm.h"
#include "Msnhnet/layers/arm/MsnhConvolution3x3s1Winograd.h"
#endif

namespace Msnhnet
{
class MsnhNet_API ConvolutionalLayer:public BaseLayer
{
public:

    ConvolutionalLayer(const int &batch, const int &steps, const int &height, const int &width, const int &channel, const int &num, const int &groups,
                       const int &kSizeX, const int &kSizeY, const int &strideX, const int &strideY, const int &dilationX, const int &dilationY, const int &paddingX, const int &paddingY, ActivationType _activation, const std::vector<float> &actParams,
                       const int &batchNorm, const float &bnEps,  const int &useBias, const int &binary, const int &xnor, const int &useBinOutput, const int &groupIndex,
                       const int &antialiasing, ConvolutionalLayer *const &shareLayer, const int &assistedExcitation, const int &deform);
    ~ConvolutionalLayer();

    int convOutHeight();
    int convOutWidth();

    int getConvWorkSpaceSize();
    int getWorkSpaceSize32();
    int getWorkSpaceSize16();

    static void  addBias(float *const &output, float *const &biases, const int &batch, const int &num, const int &whSize);
    static void  scaleBias(float *const &output, float *const &scales, const int &batch, const int &num, const int &whSize);

    void binarizeWeights(float *const &weights, const int &num, const int &wtSize, float *const &binary);
    void cpuBinarize(float *const &x, const int &xNum, float *const &binary);
    void swapBinary();

    virtual void mallocMemory();
    void forward(NetworkState &netState);
#ifdef USE_GPU
    void forwardGPU(NetworkState &netState);
#endif
    void loadAllWeigths(std::vector<float> &weights);

    virtual void saveAllWeights(const int &mainIdx, const int &branchIdx=-1, const int &branchIdx1=-1);

    void loadScales(float *const &weights, const int& len);
    void loadBias(float *const &bias, const int& len);
    void loadWeights(float *const &weights, const int& len);
    void loadRollMean(float * const &rollMean, const int &len);
    void loadRollVariance(float * const &rollVariance, const int &len);
    void loadPreluWeights(float *const &weights, const int& len); 

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

    int getNPreluWeights() const;

protected:

#ifdef USE_ARM
    void selectArmConv(); 

    float       *_sgemmWeightsPack4  =   nullptr;
    float       *_winogradWeights1   =   nullptr;
    float       *_winogradWeights2   =   nullptr;

    int         _winogradOutWidth    =   0;
    int         _winogradOutHeight   =   0;

    bool        useWinograd3x3S1     =   false;
    bool        useIm2ColSgemm       =   false;
    bool        use3x3S1             =   false;
    bool        use3x3S2             =   false;
#endif
    float       *_weights            =   nullptr;
    float       *_biases             =   nullptr;
    ConvolutionalLayer* _shareLayer  =   nullptr;

    float       *_scales             =   nullptr;
    float       *_rollMean           =   nullptr;
    float       *_rollVariance       =   nullptr;
    float       *_preluWeights       =   nullptr;

    char        *_cWeights           =   nullptr;
    float       *_binaryInputs       =   nullptr;
    float       *_binaryWeights      =   nullptr;
    float       *_meanArr            =   nullptr;
    uint32_t    *_binRePackedIn      =   nullptr;
    char        *_tBitInput          =   nullptr;
    char        *_alignBitWeights    =   nullptr;

#ifdef USE_GPU
#ifdef USE_CUDNN

    cudnnConvolutionFwdAlgo_t       _fwAlgo;

    cudnnConvolutionDescriptor_t    _convDesc;

    cudnnTensorDescriptor_t         _inputDesc;
    cudnnTensorDescriptor_t         _outputDesc;

    cudnnFilterDescriptor_t         _weightDesc;

    cudnnConvolutionFwdAlgo_t       _fwAlgo16;

    cudnnTensorDescriptor_t         _inputDesc16;
    cudnnTensorDescriptor_t         _outputDesc16;

    cudnnFilterDescriptor_t         _weightDesc16;

    float        *_gpuWeightsFp16    =   nullptr;
    float        *_gpuOutputFp16     =   nullptr;

#endif

    float       *_gpuWeights         =   nullptr;
    float       *_gpuBiases          =   nullptr;

    float       *_gpuScales          =   nullptr;
    float       *_gpuRollMean        =   nullptr;
    float       *_gpuRollVariance    =   nullptr;
    float       *_gpuPreluWeights    =   nullptr;

#endif

    int         _bitAlign            =   0;
    int         _ldaAlign            =   0;

    int         _useBias             =   1;

    int         _nPreluWeights       =   0;
    int         _nScales             =   0;
    int         _nRollMean           =   0;
    int         _nRollVariance       =   0;
    float       _bnEps               =   0.00001f;

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

