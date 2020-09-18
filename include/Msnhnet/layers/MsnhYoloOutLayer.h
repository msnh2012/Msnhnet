#ifndef MSNHYOLOOUTLAYER_H
#define MSNHYOLOOUTLAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/utils/MsnhMathUtils.h"
#include "Msnhnet/layers/MsnhYoloDef.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExVector.h"
#include "Msnhnet/utils/MsnhExport.h"

#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhYoloOutLayerGPU.h"
#include "Msnhnet/layers/cuda/MsnhYoloLayerGPU.h"
#endif

namespace Msnhnet
{
class MsnhNet_API YoloOutLayer:public BaseLayer
{
public:
    YoloOutLayer(const int &batch, const int &orgWidth, const int &orgHeight, std::vector<int> &yoloIndexes, std::vector<YoloInfo> &yoloLayersInfo,
                   const float &confThresh, const float &nmsThresh, const int &useSoftNms, const YoloType &yoloType);
    ~YoloOutLayer();

    std::vector<bool> batchHasBox;
    std::vector<std::vector<YoloBox>> finalOut;

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    static YoloBox bboxResize2Org(YoloBox &box, const Point2I &currentShape , const Point2I &orgShape);

    static YoloBox bboxResize2OrgNoPad(YoloBox &box, const Point2I &currentShape , const Point2I &orgShape);

    static std::vector<YoloBox> nms(const std::vector<YoloBox> &bboxes, const float& _nmsThresh, const bool &useSoftNms=false, const float &sigma =0.3f);

    float getConfThresh() const;

    float getNmsThresh() const;

    int getUseSoftNms() const;

    int getPixels() const;

    int getOrgHeight() const;

    int getOrgWidth() const;

    YoloType getYoloType() const;

    std::vector<int> getYoloIndexes() const;

    std::vector<YoloInfo> getYoloLayersInfo() const;

    int getYoloAllInputNum() const;

    float *getAllInput() const;

    float *getShuffleInput() const;

protected:
    float   _confThresh  = 0.6f;
    float   _nmsThresh   = 0.4f;
    int     _useSoftNms  = 0;
    int     _pixels      = 0;

    int     _orgHeight   =   0;
    int     _orgWidth    =   0;
    float em = 0.f;

    YoloType _yoloType   =   YoloType::YoloV3;

    std::vector<int>        _yoloIndexes;
    std::vector<YoloInfo>   _yoloLayersInfo;

    int     _yoloAllInputNum   =   0;      

    float   *_allInput           =   nullptr;
    float   *_shuffleInput       =   nullptr;

#ifdef USE_GPU
    float   *_allInputGpu        =   nullptr;
    float   *_shuffleInputGpu    =   nullptr;
#endif

};
}

#endif 

