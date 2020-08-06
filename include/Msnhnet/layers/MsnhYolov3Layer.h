#ifndef MSNHYOLOV3LAYER_H
#define MSNHYOLOV3LAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"
#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhYolov3LayerGPU.h"
#endif

namespace Msnhnet
{
class MsnhNet_API Yolov3Layer : public BaseLayer
{
public:
    Yolov3Layer(const int &_batch, const int &_width, const int &_height, const int &_num, const int &_orgWidth, const int &_orgHeight, const int &_classNum, const std::vector<float> &anchors);

    std::vector<float> anchors;

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    void sigmoid(float *val, const int &_num);

    void exSigmoid(float *val, const int &_width, const int &_height, const float& _ratios, const bool &addGridW);

    void aExpT(float *val, const int &_num, const float &a);

    int getClassNum() const;

    int getOrgHeight() const;

    int getOrgWidth() const;

    float getRatios() const;

protected:
    int         _classNum    =   0;

    int         _orgHeight   =   0;
    int         _orgWidth    =   0;

    float       _ratios      =   0;

};
}
#endif 

