#include "Msnhnet/layers/MsnhYolov3OutLayer.h"
namespace Msnhnet
{
Yolov3OutLayer::Yolov3OutLayer(const int &batch, const int &orgWidth, const int &orgHeight, std::vector<int> &yolov3Indexes, std::vector<Yolov3Info> &yolov3LayersInfo,
                               const float &confThresh, const float &nmsThresh, const int &useSotfNms, const YoloType &yoloType)
{
    this->_type              =   LayerType::YOLOV3_OUT;
    this->_layerName         =   "Yolov3Out       ";

    this->_yoloType          =   yoloType;

    this->_batch             =   batch;
    this->_confThresh        =   confThresh;
    this->_nmsThresh         =   nmsThresh;
    this->_useSoftNms        =   useSotfNms;

    this->_orgHeight         =   orgHeight;
    this->_orgWidth          =   orgWidth;

    this->_layerDetail.append("Yolov3out  ");
    char msg[100];

    this->_yolov3Indexes     =   yolov3Indexes;
    this->_yolov3LayersInfo  =   yolov3LayersInfo;

    for (size_t i = 0; i < yolov3Indexes.size(); ++i)
    {
        this->_width         +=   yolov3LayersInfo[i].outWidth;
        this->_height        +=   yolov3LayersInfo[i].outHeight;

#ifdef WIN32
        sprintf_s(msg, " %d", yolov3Indexes[i]);
#else
        sprintf(msg, " %d", yolov3Indexes[i]);
#endif
        this->_layerDetail.append(msg);

        this->_yolov3AllInputNum += yolov3LayersInfo[i].getOutputNum();
    }

    this->_channel           =   yolov3LayersInfo[0].outChannel/3;    

    this->_pixels            =   this->_yolov3AllInputNum / _channel; 

    this->_layerDetail.append("\n");

    if(!BaseLayer::isPreviewMode)
    {
        this->_allInput             =   new float[static_cast<size_t>(this->_yolov3AllInputNum * this->_batch)]();
#ifndef USE_GPU
        this->_shuffleInput         =   new float[static_cast<size_t>(this->_yolov3AllInputNum * this->_batch)]();
#endif
#ifdef USE_GPU
        CUDA_CHECK(cudaHostAlloc(&this->_shuffleInput, this->_yolov3AllInputNum * this->_batch * sizeof(float), cudaHostRegisterMapped));  

        this->_allInputGpu          =   Cuda::makeCudaArray(this->_allInput, this->_yolov3AllInputNum * this->_batch);
        this->_shuffleInputGpu      =   Cuda::makeCudaArray(this->_shuffleInput, this->_yolov3AllInputNum * this->_batch);
#endif
    }
}

Yolov3OutLayer::~Yolov3OutLayer()
{
    releaseArr(_allInput);
#ifndef USE_GPU
    releaseArr(_shuffleInput);
#endif

#ifdef USE_GPU
    CUDA_CHECK(cudaFreeHost(_shuffleInput));
    Cuda::freeCuda(_allInputGpu);
    Cuda::freeCuda(_shuffleInputGpu);
#endif
}

void Yolov3OutLayer::forward(NetworkState &netState)
{

    auto st = TimeUtil::startRecord();

    batchHasBox.clear();
    finalOut.clear();
    std::vector<bool> tmpBatchHasBox(static_cast<size_t>(this->_batch),false);

    int offset          =   0;
    for (int b = 0; b < this->_batch; ++b)
    {
        for (int i = 0; i < this->_yolov3Indexes.size(); ++i)
        {
            size_t index        =   static_cast<size_t>(this->_yolov3Indexes[i]);
            float *mInput       =   netState.net->layers[index]->getOutput();
            int yolov3InputNum  =   netState.net->layers[index]->getOutputNum();

            Blas::cpuCopy(yolov3InputNum, mInput, 1, this->_allInput+offset,1);

            int WxH             =   netState.net->layers[index]->getOutWidth()*netState.net->layers[index]->getOutHeight();
            int chn             =   netState.net->layers[index]->getOutChannel()/3;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int k = 0; k < 3; ++k)
            {
                for (int n = 0; n < WxH; ++n)
                {
                    for (int m = 0; m < chn; ++m)
                    {
                        this->_shuffleInput[offset + k*WxH*chn + n*chn + m] = this->_allInput[offset + k*WxH*chn+ m*WxH + n];
                    }
                }
            }

            offset              =   offset + yolov3InputNum;
        }

        std::vector<Yolov3Box> tmpBox;

        for (int i = 0; i < this->_pixels; ++i)
        {
            int ptr             =   this->_yolov3AllInputNum*b;

            if(this->_shuffleInput[ptr + i*this->_channel + 4] > this->_confThresh)
            {
                Yolov3Box box;

                box.xywhBox         =   Box::XYWHBox(this->_shuffleInput[ptr + i*this->_channel],
                                                    this->_shuffleInput[ptr + i*this->_channel + 1],
                                                    this->_shuffleInput[ptr + i*this->_channel + 2],
                                                    this->_shuffleInput[ptr + i*this->_channel + 3]);

                box.conf            =   this->_shuffleInput[ptr + i*this->_channel + 4];

                if(_yoloType == YoloType::YoloV3_NORMAL || _yoloType == YoloType::YoloV4)
                {
                    for (int j = 0; j < this->_channel - 5; ++j)
                    {
                        box.classesVal.push_back(this->_shuffleInput[ptr + i*this->_channel + 5 + j]);
                    }
                }
                else if(_yoloType == YoloType::YoloV3_ANGLE)
                {
                    for (int j = 0; j < this->_channel - 5 - 7; ++j) 

                    {
                        box.classesVal.push_back(this->_shuffleInput[ptr + i*this->_channel + 5 + j]);
                    }

                    std::vector<float> angleSplits;

                    for (int j = 0; j < 6; ++j)
                    {
                        angleSplits.push_back(this->_shuffleInput[ptr + i*this->_channel +  this->_channel - 7 + j]);
                    }

                    float regAngle    = 0.f;

                    regAngle = this->_shuffleInput[ptr + i*this->_channel +  this->_channel - 1];

                    int bestAngleIndex = static_cast<int>(ExVector::maxIndex(angleSplits));
                    box.angle = bestAngleIndex*30.f + regAngle*30 - 90.f;

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

        finalOut.push_back(nms(tmpBox, this->_nmsThresh, this->_useSoftNms));
    }

    this->batchHasBox   =   tmpBatchHasBox;

    this->_forwardTime  =   TimeUtil::getElapsedTime(st);

}

#ifdef USE_GPU
void Yolov3OutLayer::forwardGPU(NetworkState &netState)
{
    batchHasBox.clear();
    finalOut.clear();
    auto st = TimeUtil::startRecord();

    std::vector<bool> tmpBatchHasBox(static_cast<size_t>(this->_batch),false);

    int offset          =   0;

    for (int b = 0; b < this->_batch; ++b)
    {
        for (int i = 0; i < this->_yolov3Indexes.size(); ++i)
        {
            size_t index        =   static_cast<size_t>(this->_yolov3Indexes[i]);
            float *mInput       =   netState.net->layers[index]->getGpuOutput();
            int yolov3InputNum  =   netState.net->layers[index]->getOutputNum();

            CUDA_CHECK(cudaMemcpyAsync(this->_allInputGpu + offset, mInput, yolov3InputNum*sizeof(float), cudaMemcpyDeviceToDevice,Cuda::getCudaStream()));

            int WxH             =   netState.net->layers[index]->getOutWidth()*netState.net->layers[index]->getOutHeight();
            int chn             =   netState.net->layers[index]->getOutChannel()/3;

            Yolov3OutLayerGPU::shuffleData(3, WxH, chn, this->_allInputGpu + offset, this->_shuffleInputGpu + offset, 0);
            offset              =   offset + yolov3InputNum;
        }

        CUDA_CHECK(cudaMemcpy(this->_shuffleInput, this->_shuffleInputGpu, this->_yolov3AllInputNum * this->_batch*sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<Yolov3Box> tmpBox;

        for (int i = 0; i < this->_pixels; ++i)
        {
            int ptr             =   this->_yolov3AllInputNum*b;

            if(this->_shuffleInput[ptr + i*this->_channel + 4] > this->_confThresh)
            {
                Yolov3Box box;

                box.xywhBox         =   Box::XYWHBox(this->_shuffleInput[ptr + i*this->_channel],
                        this->_shuffleInput[ptr + i*this->_channel + 1],
                        this->_shuffleInput[ptr + i*this->_channel + 2],
                        this->_shuffleInput[ptr + i*this->_channel + 3]);
                box.conf            =   this->_shuffleInput[ptr + i*this->_channel + 4];

                if(_yoloType == YoloType::YoloV3_NORMAL || _yoloType == YoloType::YoloV4)
                {
                    for (int j = 0; j < this->_channel - 5; ++j)
                    {
                        box.classesVal.push_back(this->_shuffleInput[ptr + i*this->_channel + 5 + j]);
                    }
                }
                else if(_yoloType == YoloType::YoloV3_ANGLE)
                {
                    for (int j = 0; j < this->_channel - 5 - 7; ++j) 

                    {
                        box.classesVal.push_back(this->_shuffleInput[ptr + i*this->_channel + 5 + j]);
                    }

                    std::vector<float> angleSplits;

                    for (int j = 0; j < 6; ++j)
                    {
                        angleSplits.push_back(this->_shuffleInput[ptr + i*this->_channel +  this->_channel - 7 + j]);
                    }

                    float regAngle    = 0.f;

                    regAngle = this->_shuffleInput[ptr + i*this->_channel +  this->_channel - 1];

                    int bestAngleIndex = static_cast<int>(ExVector::maxIndex(angleSplits));
                    box.angle = bestAngleIndex*30.f + regAngle*30 - 90.f;

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

        finalOut.push_back(nms(tmpBox, this->_nmsThresh, this->_useSoftNms));
    }

    this->batchHasBox   =   tmpBatchHasBox;

    this->_forwardTime =   TimeUtil::getElapsedTime(st);

}
#endif

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

float Yolov3OutLayer::getConfThresh() const
{
    return _confThresh;
}

float Yolov3OutLayer::getNmsThresh() const
{
    return _nmsThresh;
}

int Yolov3OutLayer::getUseSoftNms() const
{
    return _useSoftNms;
}

int Yolov3OutLayer::getPixels() const
{
    return _pixels;
}

int Yolov3OutLayer::getOrgHeight() const
{
    return _orgHeight;
}

int Yolov3OutLayer::getOrgWidth() const
{
    return _orgWidth;
}

YoloType Yolov3OutLayer::getYoloType() const
{
    return _yoloType;
}

std::vector<int> Yolov3OutLayer::getYolov3Indexes() const
{
    return _yolov3Indexes;
}

std::vector<Yolov3Info> Yolov3OutLayer::getYolov3LayersInfo() const
{
    return _yolov3LayersInfo;
}

int Yolov3OutLayer::getYolov3AllInputNum() const
{
    return _yolov3AllInputNum;
}

float *Yolov3OutLayer::getAllInput() const
{
    return _allInput;
}

float *Yolov3OutLayer::getShuffleInput() const
{
    return _shuffleInput;
}

}
