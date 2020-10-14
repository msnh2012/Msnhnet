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
                         const int &groups, const ActivationType &activation, const std::vector<float> &actParams,const int &useBias);

    virtual void mallocMemory();
    virtual void forward(NetworkState &netState);
    virtual void loadAllWeigths(std::vector<float> &weights);
    virtual void saveAllWeights(const int &mainIdx, const int &branchIdx=-1, const int &branchIdx1=-1);

    void loadBias(float *const &bias, const int& len);
    void loadWeights(float *const &weights, const int& len);

    int deConvOutHeight();
    int deConvOutWidth();

    float *getWeights() const;

    float *getBiases() const;

    float *getColImg() const;

    int getKSizeX() const;

    int getKSizeY() const;

    int getStride() const;

    int getStrideX() const;

    int getStrideY() const;

    int getPaddingX() const;

    int getPaddingY() const;

    int getUseBias() const;

    int getNBiases() const;

    int getNWeights() const;

    int getGroups() const;

private:
    float       *_weights            =   nullptr;
    float       *_biases             =   nullptr;
    float       *_colImg             =   nullptr;

    int         _kSizeX              =   0;
    int         _kSizeY              =   0;

    int         _stride              =   0;
    int         _strideX             =   0;
    int         _strideY             =   0;

    int         _paddingX            =   0;
    int         _paddingY            =   0;

    int         _useBias             =   1;

    int         _nBiases             =   0;
    int         _nWeights            =   0;

    int         _groups              =   1;
};
}
#endif 

