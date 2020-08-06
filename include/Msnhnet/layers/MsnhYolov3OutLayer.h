#ifndef MSNHYOLOV3OUTLAYER_H
#define MSNHYOLOV3OUTLAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/utils/MsnhMathUtils.h"
#include "Msnhnet/layers/MsnhYolov3Def.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExVector.h"
#include "Msnhnet/utils/MsnhExport.h"

#ifdef USE_GPU
#include "Msnhnet/layers/cuda/MsnhYolov3OutLayerGPU.h"
#include "Msnhnet/layers/cuda/MsnhYolov3LayerGPU.h"
#endif

namespace Msnhnet
{
class MsnhNet_API Yolov3OutLayer:public BaseLayer
{
public:
    Yolov3OutLayer(const int &_batch, const int &_orgWidth, const int &_orgHeight, std::vector<int> &_yolov3Indexes, std::vector<Yolov3Info> &_yolov3LayersInfo,
                   const float &_confThresh, const float &_nmsThresh, const int &_useSoftNms, const YoloType &_yoloType);
    ~Yolov3OutLayer();

    std::vector<bool> batchHasBox;
    std::vector<std::vector<Yolov3Box>> finalOut;

    virtual void forward(NetworkState &netState);

#ifdef USE_GPU
    virtual void forwardGPU(NetworkState &netState);
#endif

    static Yolov3Box bboxResize2org(Yolov3Box &box, const Point2I &currentShape , const Point2I &orgShape);

    static std::vector<Yolov3Box> nms(const std::vector<Yolov3Box> &bboxes, const float& _nmsThresh, const bool &_useSoftNms=false, const float &sigma =0.3f);

    float getConfThresh() const;

    float getNmsThresh() const;

    int getUseSoftNms() const;

    int getPixels() const;

    int getOrgHeight() const;

    int getOrgWidth() const;

    YoloType getYoloType() const;

    std::vector<int> getYolov3Indexes() const;

    std::vector<Yolov3Info> getYolov3LayersInfo() const;

    int getYolov3AllInputNum() const;

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

    YoloType _yoloType   =   YoloType::YoloV3_NORMAL;

    std::vector<int>        _yolov3Indexes;
    std::vector<Yolov3Info> _yolov3LayersInfo;

    int     _yolov3AllInputNum   =   0;      

    float   *_allInput           =   nullptr;
    float   *_shuffleInput       =   nullptr;

#ifdef USE_GPU
    float   *_allInputGpu        =   nullptr;
    float   *_shuffleInputGpu    =   nullptr;
#endif

};
}

#endif 

