#ifndef MSNHYOLOLAYER_H
#define MSNHYOLOLAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/layers/MsnhYoloDef.h"
#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhYoloLayerGPU.h"
#endif

namespace Msnhnet
{
class MsnhNet_API YoloLayer : public BaseLayer
{
public:
    YoloLayer(const int &batch, const int &width, const int &height, const int &num, const int &orgWidth, const int &orgHeight, const int &classNum,
              const std::vector<float> &anchors, const YoloType &yoloType);

    std::vector<float> anchors;

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    void sigmoid(float *val, const int &num);

    void exSigmoid(float *val, const int &width, const int &height, const float& ratios, const bool &addGridW);

    void exSigmoidV5(float *val, const int &width, const int &height, const float& ratios, const bool &addGridW);

    void aExpT(float *val, const int &num, const float &a);

    void aPowSigmoid(float *val, const int &num, const float &a);

    int getClassNum() const;

    int getOrgHeight() const;

    int getOrgWidth() const;

    float getRatios() const;

    YoloType getYoloType() const;

protected:
    int         _classNum    =   0;

    int         _orgHeight   =   0;
    int         _orgWidth    =   0;

    float       _ratios      =   0;
    YoloType    _yoloType    =   YoloType::YoloV3;

};
}
#endif 

