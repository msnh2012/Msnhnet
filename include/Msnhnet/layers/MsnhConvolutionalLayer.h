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

    ConvolutionalLayer(const int &batch, const int &steps, const int &height, const int &width, const int &channel, const int &num, const int &groups,
                      const int &kSizeX, const int &kSizeY, const int &strideX, const int &strideY, const int &dilationX, const int &dilationY, const int &paddingX, const int &paddingY, ActivationType activation, const std::vector<float> &actParams,
                      const int &batchNorm,  const int &useBias, const int &binary, const int &xnor, const int &useBinOutput, const int &groupIndex,
                      const int &antialiasing, ConvolutionalLayer *const &shareLayer, const int &assistedExcitation, const int &deform);
    ~ConvolutionalLayer();

    float       *weights            =   nullptr;
    float       *biases             =   nullptr;

    ConvolutionalLayer* shareLayer  =   nullptr;

    float       *scales             =   nullptr;
    float       *rollMean           =   nullptr;
    float       *rollVariance       =   nullptr;

    char        *cWeights           =   nullptr;
    float       *binaryInputs       =   nullptr;
    float       *binaryWeights      =   nullptr;
    float       *activationInput    =   nullptr;
    float       *meanArr            =   nullptr;
    uint32_t    *binRePackedIn      =   nullptr;
    char        *tBitInput          =   nullptr;
    char        *alignBitWeights    =   nullptr;

    int         bitAlign            =   0;
    int         ldaAlign            =   0;

    int         useBias             =   1;

    int         nScales             =   0;
    int         nRollMean           =   0;
    int         nRollVariance       =   0;

    int         nBiases             =   0;
    int         nWeights            =   0;
    int         groups              =   0;
    int         groupIndex          =   0;

    int         xnor                =   0;
    int         binary              =   0;
    int         useBinOutput        =   0;
    int         steps               =   0;

    int         antialiasing        =   0;
    int         assistedExcite      =   0;

    int         kSizeX              =   0;
    int         kSizeY              =   0;
    int         stride              =   0;
    int         strideX             =   0;
    int         strideY             =   0;
    int         paddingX            =   0;
    int         paddingY            =   0;
    int         dilationX           =   0;
    int         dilationY           =   0;
    int         batchNorm           =   0;

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

    void forward(NetworkState &netState);
    void loadAllWeigths(std::vector<float> &weights);

    void loadScales(float *const &weights, const int& len);
    void loadBias(float *const &bias, const int& len);
    void loadWeights(float *const &weights, const int& len);
    void loadRollMean(float * const &rollMean, const int &len);
    void loadRollVariance(float * const &rollVariance, const int &len);

};
}

#endif 
