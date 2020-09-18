#include "Msnhnet/layers/MsnhYoloOutLayer.h"
namespace Msnhnet
{
YoloOutLayer::YoloOutLayer(const int &batch, const int &orgWidth, const int &orgHeight, std::vector<int> &yoloIndexes, std::vector<YoloInfo> &yoloLayersInfo,
                           const float &confThresh, const float &nmsThresh, const int &useSotfNms, const YoloType &yoloType)
{
    this->_type              =   LayerType::YOLO_OUT;
    this->_layerName         =   "YoloOut         ";

    this->_yoloType          =   yoloType;

    this->_batch             =   batch;
    this->_confThresh        =   confThresh;
    this->_nmsThresh         =   nmsThresh;
    this->_useSoftNms        =   useSotfNms;

    this->_orgHeight         =   orgHeight;
    this->_orgWidth          =   orgWidth;

    this->_layerDetail.append("YoloOut    ");
    char msg[100];

    this->_yoloIndexes     =   yoloIndexes;
    this->_yoloLayersInfo  =   yoloLayersInfo;

    for (size_t i = 0; i < yoloIndexes.size(); ++i)
    {
        this->_width         +=   yoloLayersInfo[i].outWidth;
        this->_height        +=   yoloLayersInfo[i].outHeight;

#ifdef WIN32
        sprintf_s(msg, " %d", yoloIndexes[i]);
#else
        sprintf(msg, " %d", yoloIndexes[i]);
#endif
        this->_layerDetail.append(msg);

        this->_yoloAllInputNum += yoloLayersInfo[i].getOutputNum();
    }

    this->_channel           =   yoloLayersInfo[0].outChannel/3;    

    this->_pixels            =   this->_yoloAllInputNum / _channel; 

    this->_layerDetail.append("\n");

    if(!BaseLayer::isPreviewMode)
    {
        this->_allInput             =   new float[static_cast<size_t>(this->_yoloAllInputNum * this->_batch)]();
#ifndef USE_GPU
        this->_shuffleInput         =   new float[static_cast<size_t>(this->_yoloAllInputNum * this->_batch)]();
#endif
#ifdef USE_GPU
        CUDA_CHECK(cudaHostAlloc(&this->_shuffleInput, this->_yoloAllInputNum * this->_batch * sizeof(float), cudaHostRegisterMapped));  

        this->_allInputGpu          =   Cuda::makeCudaArray(this->_allInput, this->_yoloAllInputNum * this->_batch);
        this->_shuffleInputGpu      =   Cuda::makeCudaArray(this->_shuffleInput, this->_yoloAllInputNum * this->_batch);
#endif
    }
    this->_memReUse = 0;
}

YoloOutLayer::~YoloOutLayer()
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

void YoloOutLayer::forward(NetworkState &netState)
{

    auto st = TimeUtil::startRecord();

    batchHasBox.clear();
    finalOut.clear();
    std::vector<bool> tmpBatchHasBox(static_cast<size_t>(this->_batch),false);

    int offset          =   0;
    for (int b = 0; b < this->_batch; ++b)
    {
        for (int i = 0; i < this->_yoloIndexes.size(); ++i)
        {
            size_t index        =   static_cast<size_t>(this->_yoloIndexes[i]);
            float *mInput       =   netState.net->layers[index]->getOutput();
            int yoloInputNum  =   netState.net->layers[index]->getOutputNum();

            Blas::cpuCopy(yoloInputNum, mInput, 1, this->_allInput+offset,1);

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

            offset              =   offset + yoloInputNum;
        }

        std::vector<YoloBox> tmpBox;

        for (int i = 0; i < this->_pixels; ++i)
        {
            int ptr             =   this->_yoloAllInputNum*b;

            if(this->_shuffleInput[ptr + i*this->_channel + 4] > this->_confThresh)
            {
                YoloBox box;

                box.xywhBox         =   Box::XYWHBox(this->_shuffleInput[ptr + i*this->_channel],
                        this->_shuffleInput[ptr + i*this->_channel + 1],
                        this->_shuffleInput[ptr + i*this->_channel + 2],
                        this->_shuffleInput[ptr + i*this->_channel + 3]);

                box.conf            =   this->_shuffleInput[ptr + i*this->_channel + 4];

                if(_yoloType == YoloType::YoloV3 || _yoloType == YoloType::YoloV4 || _yoloType == YoloType::YoloV5)
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

            std::vector<YoloBox> tmpBox1 = tmpBox;

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
void YoloOutLayer::forwardGPU(NetworkState &netState)
{
    batchHasBox.clear();
    finalOut.clear();
    auto st = TimeUtil::startRecord();

    std::vector<bool> tmpBatchHasBox(static_cast<size_t>(this->_batch),false);

    int offset          =   0;

    for (int b = 0; b < this->_batch; ++b)
    {
        for (int i = 0; i < this->_yoloIndexes.size(); ++i)
        {
            size_t index        =   static_cast<size_t>(this->_yoloIndexes[i]);
            float *mInput       =   netState.net->layers[index]->getGpuOutput();
            int yoloInputNum    =   netState.net->layers[index]->getOutputNum();

            CUDA_CHECK(cudaMemcpyAsync(this->_allInputGpu + offset, mInput, yoloInputNum*sizeof(float), cudaMemcpyDeviceToDevice,Cuda::getCudaStream()));

            int WxH             =   netState.net->layers[index]->getOutWidth()*netState.net->layers[index]->getOutHeight();
            int chn             =   netState.net->layers[index]->getOutChannel()/3;

            YoloOutLayerGPU::shuffleData(3, WxH, chn, this->_allInputGpu + offset, this->_shuffleInputGpu + offset, 0);
            offset              =   offset + yoloInputNum;
        }

        CUDA_CHECK(cudaMemcpy(this->_shuffleInput, this->_shuffleInputGpu, this->_yoloAllInputNum * this->_batch*sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<YoloBox> tmpBox;

        for (int i = 0; i < this->_pixels; ++i)
        {
            int ptr             =   this->_yoloAllInputNum*b;

            if(this->_shuffleInput[ptr + i*this->_channel + 4] > this->_confThresh)
            {
                YoloBox box;

                box.xywhBox         =   Box::XYWHBox(this->_shuffleInput[ptr + i*this->_channel],
                        this->_shuffleInput[ptr + i*this->_channel + 1],
                        this->_shuffleInput[ptr + i*this->_channel + 2],
                        this->_shuffleInput[ptr + i*this->_channel + 3]);
                box.conf            =   this->_shuffleInput[ptr + i*this->_channel + 4];

                if(_yoloType == YoloType::YoloV3 || _yoloType == YoloType::YoloV4 || _yoloType == YoloType::YoloV5)
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

            std::vector<YoloBox> tmpBox1 = tmpBox;

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

YoloBox YoloOutLayer::bboxResize2Org( YoloBox &box, const Point2I &currentShape, const Point2I &orgShape)
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

YoloBox YoloOutLayer::bboxResize2OrgNoPad(YoloBox &box, const Point2I &currentShape, const Point2I &orgShape)
{

    int orgW    =   orgShape.x;
    int orgH    =   orgShape.y;

    int curW    =   currentShape.x;
    int curH    =   currentShape.y;

    float scaledRatioW   =   1.0f * curW / orgW;
    float scaledRatioH   =   1.0f * curH / orgH;

    box.xywhBox.x       =   box.xywhBox.x / scaledRatioW;
    box.xywhBox.w       =   box.xywhBox.w / scaledRatioW;
    box.xywhBox.y       =   box.xywhBox.y / scaledRatioH;
    box.xywhBox.h       =   box.xywhBox.h / scaledRatioH;

    return box;
}

std::vector<YoloBox> YoloOutLayer::nms(const std::vector<YoloBox> &bboxes, const float &nmsThresh, const bool &useSoftNms, const float &sigma)
{

    std::vector<YoloBox> bestBoxes;

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
        std::vector<YoloBox> classIBboxes;

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

            YoloBox bestIBox  =   classIBboxes[bestIndex];

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

float YoloOutLayer::getConfThresh() const
{
    return _confThresh;
}

float YoloOutLayer::getNmsThresh() const
{
    return _nmsThresh;
}

int YoloOutLayer::getUseSoftNms() const
{
    return _useSoftNms;
}

int YoloOutLayer::getPixels() const
{
    return _pixels;
}

int YoloOutLayer::getOrgHeight() const
{
    return _orgHeight;
}

int YoloOutLayer::getOrgWidth() const
{
    return _orgWidth;
}

YoloType YoloOutLayer::getYoloType() const
{
    return _yoloType;
}

std::vector<int> YoloOutLayer::getYoloIndexes() const
{
    return _yoloIndexes;
}

std::vector<YoloInfo> YoloOutLayer::getYoloLayersInfo() const
{
    return _yoloLayersInfo;
}

int YoloOutLayer::getYoloAllInputNum() const
{
    return _yoloAllInputNum;
}

float *YoloOutLayer::getAllInput() const
{
    return _allInput;
}

float *YoloOutLayer::getShuffleInput() const
{
    return _shuffleInput;
}

}
