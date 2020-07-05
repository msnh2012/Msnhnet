#ifndef MSNHYOLOV3OUTLAYER_H
#define MSNHYOLOV3OUTLAYER_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhBlas.h"
#include "Msnhnet/utils/MsnhMathUtils.h"
#include "Msnhnet/layers/MsnhYolov3Def.h"
#include "Msnhnet/layers/MsnhBaseLayer.h"
#include "Msnhnet/utils/MsnhExVector.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API Yolov3OutLayer:public BaseLayer
{
public:
    Yolov3OutLayer(const int &batch, const int &orgWidth, const int &orgHeight, std::vector<int> &yolov3Indexes, std::vector<Yolov3Info> &yolov3LayersInfo,
                   const float &confThresh, const float &nmsThresh, const int &useSoftNms);
    ~Yolov3OutLayer();
    float   confThresh  = 0.6f;
    float   nmsThresh   = 0.4f;
    int     useSoftNms  = 0;
    int     pixels      = 0;

   int     orgHeight   =   0;
    int     orgWidth    =   0;

   std::vector<int> yolov3Indexes;
    std::vector<Yolov3Info> yolov3LayersInfo;

   int     yolov3AllInputNum   =   0;      

   float   *allInput           =   nullptr;
    float   *shuffleInput       =   nullptr;

   std::vector<bool> batchHasBox;
    std::vector<std::vector<Yolov3Box>> finalOut;

   virtual void forward(NetworkState &netState);

   static Yolov3Box bboxResize2org(Yolov3Box &box, const Point2I &currentShape , const Point2I &orgShape);

   static std::vector<Yolov3Box> nms(const std::vector<Yolov3Box> &bboxes, const float& nmsThresh, const bool &useSoftNms=false, const float &sigma =0.3f);
};
}

#endif 

