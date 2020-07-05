#include "Msnhnet/layers/MsnhYolov3OutLayer.h"
namespace Msnhnet
{
Yolov3OutLayer::Yolov3OutLayer(const int &batch, const int &orgWidth, const int &orgHeight, std::vector<int> &yolov3Indexes, std::vector<Yolov3Info> &yolov3LayersInfo,
                               const float &confThresh, const float &nmsThresh, const int &useSotfNms)
{
    this->type              =   LayerType::YOLOV3_OUT;
    this->layerName         =   "Yolov3Out       ";

   this->batch             =   batch;
    this->confThresh        =   confThresh;
    this->nmsThresh         =   nmsThresh;
    this->useSoftNms        =   useSotfNms;

   this->orgHeight         =   orgHeight;
    this->orgWidth          =   orgWidth;

   this->layerDetail.append("yolov3out  ");
    char msg[100];

   this->yolov3Indexes     =   yolov3Indexes;
    this->yolov3LayersInfo  =   yolov3LayersInfo;

   for (size_t i = 0; i < yolov3Indexes.size(); ++i)
    {
        this->width         +=   yolov3LayersInfo[i].outWidth;
        this->height        +=   yolov3LayersInfo[i].outHeight;

#ifdef WIN32
        sprintf_s(msg, " %d", yolov3Indexes[i]);
#else
        sprintf(msg, " %d", yolov3Indexes[i]);
#endif
        this->layerDetail.append(msg);

       this->yolov3AllInputNum += yolov3LayersInfo[i].getOutputNum();
    }

   this->channel           =   yolov3LayersInfo[0].outChannel/3;    

   this->pixels            =   this->yolov3AllInputNum / channel; 

   this->layerDetail.append("\n");

   if(!BaseLayer::isPreviewMode)
    {
        this->allInput             =   new float[static_cast<size_t>(this->yolov3AllInputNum * this->batch)]();
        this->shuffleInput         =   new float[static_cast<size_t>(this->yolov3AllInputNum * this->batch)]();
    }
}

Yolov3OutLayer::~Yolov3OutLayer()
{
    releaseArr(allInput);
    releaseArr(shuffleInput);
}

void Yolov3OutLayer::forward(NetworkState &netState)
{
    batchHasBox.clear();
    finalOut.clear();
    auto st = std::chrono::system_clock::now();
    std::vector<bool> tmpBatchHasBox(static_cast<size_t>(this->batch),false);

   int offset          =   0;
    for (int b = 0; b < this->batch; ++b)
    {
        for (size_t i = 0; i < this->yolov3Indexes.size(); ++i)
        {
            size_t index        =   static_cast<size_t>(this->yolov3Indexes[i]);
            float *mInput       =   netState.net->layers[index]->output;
            int yolov3InputNum  =   netState.net->layers[index]->outputNum;

           Blas::cpuCopy(yolov3InputNum, mInput, 1, this->allInput+offset,1);

           int WxH             =   netState.net->layers[index]->outWidth*netState.net->layers[index]->outHeight;
            int chn             =   netState.net->layers[index]->outChannel/3;

           for (int k = 0; k < 3; ++k)
            {
                for (int n = 0; n < WxH; ++n)
                {
                    for (int m = 0; m < chn; ++m)
                    {
                        this->shuffleInput[offset + k*WxH*chn + n*chn + m] = this->allInput[offset + k*WxH*chn+ m*WxH + n];
                    }
                }
            }

           offset              =   offset + yolov3InputNum;
        }

       std::vector<Yolov3Box> tmpBox;

       for (int i = 0; i < this->pixels; ++i)
        {
            int ptr             =   this->yolov3AllInputNum*b;

           if(this->shuffleInput[ptr + i*this->channel + 4] > this->confThresh)
            {
                Yolov3Box box;

               box.xywhBox         =   Box::XYWHBox(this->shuffleInput[ptr + i*this->channel],
                        this->shuffleInput[ptr + i*this->channel + 1],
                        this->shuffleInput[ptr + i*this->channel + 2],
                        this->shuffleInput[ptr + i*this->channel + 3]);
                box.conf            =   this->shuffleInput[ptr + i*this->channel + 4];

               for (int j = 0; j < this->channel - 5; ++j)
                {
                    box.classesVal.push_back(this->shuffleInput[ptr + i*this->channel + 5 + j]);
                }

               box.bestClsConf     =   ExVector::max<float>(box.classesVal);
                box.bestClsIdx      =   static_cast<int>(ExVector::maxIndex(box.classesVal));

               tmpBatchHasBox[b]   =   true;
                tmpBox.push_back(box);
            }
        }

       if(tmpBatchHasBox[b]) 

       {

           std::vector<float> confs;
            for (int i = 0; i < tmpBox.size(); ++i)
            {
                float conf = tmpBox[i].conf * tmpBox[i].bestClsConf;
                confs.push_back(conf);
            }

           std::vector<int> confIndex = ExVector::argsort(confs, true);

           std::vector<Yolov3Box> tmpBox1 = tmpBox;

           for (size_t i = 0; i < confIndex.size(); ++i)
            {
                tmpBox[i]  =   tmpBox1[static_cast<size_t>(confIndex[i])];
            }
        }

       finalOut.push_back(nms(tmpBox, this->nmsThresh, this->useSoftNms));
    }

   this->batchHasBox   =   tmpBatchHasBox;

   auto so = std::chrono::system_clock::now();
    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

Yolov3Box Yolov3OutLayer::bboxResize2org( Yolov3Box &box, const Point2I &currentShape, const Point2I &orgShape)
{

   /*
         w > h       w < h
        =padwh=     =padxy=
        0 0 0 0     0 x x 0
        x x x x     0 x x 0
        x x x x     0 x x 0
        0 0 0 0     0 x x 0
       */

   int orgW    =   orgShape.x;
    int orgH    =   orgShape.y;

   int curW    =   currentShape.x;
    int curH    =   currentShape.y;

   if(orgW > orgH)
    {
        float scaledRatio   =   1.0f * curW / orgW;
        int   padUp         =   static_cast<int>((curH - orgH * scaledRatio)/2);
        box.xywhBox.x       =   box.xywhBox.x / scaledRatio;
        box.xywhBox.w       =   box.xywhBox.w / scaledRatio;
        box.xywhBox.y       =   (box.xywhBox.y-padUp) / scaledRatio;
        box.xywhBox.h       =   box.xywhBox.h / scaledRatio;
    }
    else
    {
        float scaledRatio   =   1.0f * curH / orgH;
        int   padLeft       =   static_cast<int>((curW - orgW * scaledRatio)/2);
        box.xywhBox.x       =   (box.xywhBox.x - padLeft) / scaledRatio;
        box.xywhBox.w       =   box.xywhBox.w / scaledRatio;
        box.xywhBox.y       =   box.xywhBox.y / scaledRatio;
        box.xywhBox.h       =   box.xywhBox.h / scaledRatio;

   }

   return box;
}

std::vector<Yolov3Box> Yolov3OutLayer::nms(const std::vector<Yolov3Box> &bboxes, const float &nmsThresh, const bool &useSoftNms, const float &sigma)
{

   std::vector<Yolov3Box> bestBoxes;

   std::vector<int> classes;
    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        int index = bboxes[i].bestClsIdx;
        if(!ExVector::contains<int>(classes,index))
        {
            classes.push_back(index);
        }
    }

   for (size_t i = 0; i < classes.size(); ++i)
    {
        std::vector<Yolov3Box> classIBboxes;

       for (size_t j = 0; j < bboxes.size(); ++j)
        {
            if(bboxes[j].bestClsIdx == classes[i])
            {
                classIBboxes.push_back(bboxes[j]);
            }
        }

       if(classIBboxes.size() == 1)
        {
            bestBoxes.push_back(classIBboxes[0]);
            continue;
        }

       while(classIBboxes.size()>0)
        {
            size_t bestIndex   =   0;
            float bestConf  =   -FLT_MAX;
            for (size_t j = 0; j < classIBboxes.size(); ++j)
            {
                if(classIBboxes[j].conf > bestConf)
                {
                    bestIndex   =   j;
                    bestConf    =   classIBboxes[j].conf;
                }
            }

           Yolov3Box bestIBox  =   classIBboxes[bestIndex];

           bestBoxes.push_back(bestIBox);

           classIBboxes.erase(classIBboxes.begin()+bestIndex);

           for (size_t j = 0; j < classIBboxes.size(); ++j)
            {
                float iou       =   Box::iou(classIBboxes[j].xywhBox,bestIBox.xywhBox);

               if(useSoftNms == 1)
                {
                    float weight =  expf(-(1.f*iou*iou/sigma));
                    float conf   =  classIBboxes[j].conf*weight;

                   if(conf <= 0)
                    {
                        classIBboxes.erase(classIBboxes.begin() + (j));
                        j--;
                    }
                }
                else
                {
                    if(iou > nmsThresh)
                    {
                        classIBboxes.erase(classIBboxes.begin() + (j));
                        j--;
                    }
                }
            }

       }
    }
    return bestBoxes;
}

}
