#ifndef MSNHYOLOV3LAYER_H
#define MSNHYOLOV3LAYER_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API Yolov3Layer : public BaseLayer
{
public:
    Yolov3Layer(const int &batch, const int &width, const int &height, const int &num, const int &orgWidth, const int &orgHeight, const int &classNum, const std::vector<float> &anchors);

    int         classNum    =   0;

    int         orgHeight   =   0;
    int         orgWidth    =   0;

    float       ratios      =   0;

    std::vector<float> anchors;

    virtual void forward(NetworkState &netState);

    void sigmoid(float *val, const int &num);
    void exSigmoid(float *val, const int &width, const int &height, const float& ratios, const bool &addGridW);
    void aExpT(float *val, const int &num, const float &a);

};
}
#endif 

