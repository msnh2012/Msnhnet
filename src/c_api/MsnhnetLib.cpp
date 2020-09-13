#include "MsnhnetLib.h"

static std::unique_ptr<Msnhnet::NetBuilder> net;
int initMsnhnet()
{
    net.reset(new Msnhnet::NetBuilder());
    return 1;
}

int buildMsnhnet(char **msg, const char *msnhnet, const char *msnhbin, int useFp16, int useCudaOnly)
{
    try
    {
#ifdef USE_GPU
        net->setUseFp16((useFp16>0));
        net->setForceUseCuda((useCudaOnly>0));
#endif
        net->buildNetFromMsnhNet(msnhnet);
        net->loadWeightsFromMsnhBin(msnhbin);
    }
    catch(Msnhnet::Exception ex)
    {
        std::string tmpMsg;
        tmpMsg = tmpMsg + std::string(ex.what()) + " " + ex.getErrFile() + " " + std::to_string(ex.getErrLine()) + " " + ex.getErrFun() + " \r\n";
        std::vector<char> chr{tmpMsg.data(),tmpMsg.data()+tmpMsg.length()};
        *msg = chr.data();
        return -1;
    }

    return 1;
}

int runClassifyFile(char **msg, const char *imagePath, int *bestIndex, PredDataType preDataType, int runGPU, const float* mean, const float *std)
{
    try
    {
        int sizeX = 0;
        int sizeY = 0;
        Msnhnet::Point2I inputSize = net->getInputSize();
        sizeX = inputSize.x;
        sizeY = inputSize.y;

        std::vector<float> img;

        switch (preDataType)
        {
        case PRE_DATA_NONE:
            throw Msnhnet::Exception(1,"preDataType can't be none", __FILE__, __LINE__, __FUNCTION__);
            break;
        case PRE_DATA_FC32_C1:
            img = Msnhnet::OpencvUtil::getImgDataF32C1(imagePath, {sizeX, sizeY});
            break;
        case PRE_DATA_FC32_C3:
            img = Msnhnet::OpencvUtil::getImgDataF32C3(imagePath, {sizeX, sizeY});
            break;
        case PRE_DATA_GOOGLENET_FC3:
            img = Msnhnet::OpencvUtil::getGoogLenetF32C3(imagePath, {sizeX, sizeY});
            break;
        case PRE_DATA_PADDING_ZERO_FC3:
            img = Msnhnet::OpencvUtil::getPaddingZeroF32C3(imagePath, {sizeX, sizeY});
            break;
        case PRE_DATA_CAFFE_FC3:
            img = Msnhnet::OpencvUtil::getCaffeModeF32C3(imagePath, {sizeX, sizeY});
            break;
        case PRE_DATA_TRANSFORMED_FC3:
            if(mean == nullptr || std == nullptr)
            {
                throw Msnhnet::Exception(1,"mean or std can't be null", __FILE__, __LINE__, __FUNCTION__);
            }
            img = Msnhnet::OpencvUtil::getTransformedF32C3(imagePath, {sizeX, sizeY},cv::Scalar(mean[0], mean[1], mean[2]),cv::Scalar(std[0], std[1], std[2]));
            break;
        }

        std::vector<float> result;

#ifdef USE_GPU
        if(runGPU>0)
        {
            result =  net->runClassifyGPU(img);
        }
        else
        {
            result = net->runClassify(img);
        }
#else
        if(runGPU>0)
        {
            throw Msnhnet::Exception(1,"Msnhnet is not compiled with GPU!",__FILE__,__LINE__,__FUNCTION__);
        }
        else
        {
            result = net->runClassify(img);
        }
#endif

        *bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
    }
    catch(Msnhnet::Exception ex)
    {
        std::string tmpMsg;
        tmpMsg = tmpMsg + std::string(ex.what()) + " " + ex.getErrFile() + " " + std::to_string(ex.getErrLine()) + " " + ex.getErrFun() + " \r\n";
        std::vector<char> chr{tmpMsg.data(),tmpMsg.data()+tmpMsg.length()};
        *msg = chr.data();
        return -1;
    }
    return 1;
}

int dispose()
{
    net.reset();
    return 1;
}

int runClassifyList(char **msg, unsigned char *imageData, const int width, const int height, const int channel,
                    int *bestIndex, PredDataType preDataType, const int swapRGB,  int runGPU, const float *mean, const float *std)
{
    try
    {
        if(channel != 1 && channel !=3 )
        {
            throw Msnhnet::Exception(1,"channel must be 1 or 3", __FILE__, __LINE__, __FUNCTION__);
        }

        if(channel == 1 && preDataType!=PRE_DATA_FC32_C1)
        {
            throw Msnhnet::Exception(1,"channel 1 must preprocess with F32_C1", __FILE__, __LINE__, __FUNCTION__);
        }

        if(channel == 3 && preDataType==PRE_DATA_FC32_C1)
        {
            throw Msnhnet::Exception(1,"channel 1 can't preprocess with F32_C1", __FILE__, __LINE__, __FUNCTION__);
        }

        cv::Mat inMat(height, width,  ((channel)==1)?CV_8UC1:CV_8UC3,imageData,0);

        if(swapRGB)
        {
            cv::cvtColor(inMat,inMat,cv::COLOR_RGB2BGR);
        }

        int sizeX = 0;
        int sizeY = 0;
        Msnhnet::Point2I inputSize = net->getInputSize();
        sizeX = inputSize.x;
        sizeY = inputSize.y;

        std::vector<float> img;

        switch (preDataType)
        {
        case PRE_DATA_NONE:
            throw Msnhnet::Exception(1,"preDataType can't be none", __FILE__, __LINE__, __FUNCTION__);
            break;
        case PRE_DATA_FC32_C1:
            img = Msnhnet::OpencvUtil::getImgDataF32C1(inMat, {sizeX, sizeY});
            break;
        case PRE_DATA_FC32_C3:
            img = Msnhnet::OpencvUtil::getImgDataF32C3(inMat, {sizeX, sizeY});
            break;
        case PRE_DATA_GOOGLENET_FC3:
            img = Msnhnet::OpencvUtil::getGoogLenetF32C3(inMat, {sizeX, sizeY});
            break;
        case PRE_DATA_PADDING_ZERO_FC3:
            img = Msnhnet::OpencvUtil::getPaddingZeroF32C3(inMat, {sizeX, sizeY});
            break;
        case PRE_DATA_CAFFE_FC3:
            img = Msnhnet::OpencvUtil::getCaffeModeF32C3(inMat, {sizeX, sizeY});
            break;
        case PRE_DATA_TRANSFORMED_FC3:
            if(mean == nullptr || std == nullptr)
            {
                throw Msnhnet::Exception(1,"mean or std can't be null", __FILE__, __LINE__, __FUNCTION__);
            }
            img = Msnhnet::OpencvUtil::getTransformedF32C3(inMat, {sizeX, sizeY},cv::Scalar(mean[0], mean[1], mean[2]),cv::Scalar(std[0], std[1], std[2]));
            break;
        }

        std::vector<float> result;

#ifdef USE_GPU
        if(runGPU>0)
        {
            result =  net->runClassifyGPU(img);
        }
        else
        {
            result = net->runClassify(img);
        }
#else
        if(runGPU>0)
        {
            throw Msnhnet::Exception(1,"Msnhnet is not compiled with GPU!",__FILE__,__LINE__,__FUNCTION__);
        }
        else
        {
            result = net->runClassify(img);
        }
#endif

        *bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
    }
    catch(Msnhnet::Exception ex)
    {
        std::string tmpMsg;
        tmpMsg = tmpMsg + std::string(ex.what()) + " " + ex.getErrFile() + " " + std::to_string(ex.getErrLine()) + " " + ex.getErrFun() + " \r\n";
        std::vector<char> chr{tmpMsg.data(),tmpMsg.data()+tmpMsg.length()};
        *msg = chr.data();
        return -1;
    }
    return 1;
}

int withGPU(int *GPU)
{
#ifdef USE_GPU
    *GPU = 1;
    return 1;
#else
    *GPU = 0;
    return 0;
#endif
}

int withCUDNN(int *CUDNN)
{
#ifdef USE_CUDNN
    *CUDNN = 1;
    return 1;
#else
    *CUDNN = 0;
    return 0;
#endif
}

int getCpuForwardTime(float *time)
{
    *time = net->getInferenceTime();
    return 1;
}

int getGpuForwardTime(float *time)
{
#ifdef USE_GPU
    *time = net->getGpuInferenceTime();
    return 1;
#else
    return -1;
#endif
}

int runClassifyNoPred(char **msg, const float *data, const int len, int *bestIndex, int runGPU)
{
    try
    {
        Msnhnet::Point2I point = net->getInputSize();
        int channels           = net->getInputChannel();
        int inLen              = point.x*point.y*channels;
        if(len != inLen)
        {
            throw Msnhnet::Exception(1,"Input data num err, needed : " + std::to_string(inLen) + ". given : " + std::to_string(len), __FILE__, __LINE__, __FUNCTION__);
        }
        std::vector<float> img{data,data + inLen};

        std::vector<float> result;

#ifdef USE_GPU
        if(runGPU>0)
        {
            result =  net->runClassifyGPU(img);
        }
        else
        {
            result = net->runClassify(img);
        }
#else
        if(runGPU>0)
        {
            throw Msnhnet::Exception(1,"Msnhnet is not compiled with GPU!",__FILE__,__LINE__,__FUNCTION__);
        }
        else
        {
            result = net->runClassify(img);
        }
#endif

        *bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
    }
    catch(Msnhnet::Exception ex)
    {
        std::string tmpMsg;
        tmpMsg = tmpMsg + std::string(ex.what()) + " " + ex.getErrFile() + " " + std::to_string(ex.getErrLine()) + " " + ex.getErrFun() + " \r\n";
        std::vector<char> chr{tmpMsg.data(),tmpMsg.data()+tmpMsg.length()};
        *msg = chr.data();
        return -1;
    }
    return 1;
}

int getInputDim(int *width, int *heigth, int *channel)
{
    Msnhnet::Point2I wh = net->getInputSize();
    int ch              = net->getInputChannel();

    *width              = wh.x;
    *heigth             = wh.y;
    *channel            = ch;
    return 1;
}

int runYoloFile(char **msg, const char *imagePath, BBoxContainer *bboxContainer, int *detectedNum, const int runGPU)
{
    try
    {
        int sizeX = 0;
        int sizeY = 0;
        Msnhnet::Point2I inputSize = net->getInputSize();
        sizeX = inputSize.x;
        sizeY = inputSize.y;

        std::vector<float> img = Msnhnet::OpencvUtil::getPaddingZeroF32C3(imagePath,cv::Size(sizeX,sizeY));

        std::vector<std::vector<Msnhnet::YoloBox>> result;

#ifdef USE_GPU
        if(runGPU>0)
        {
            result =  net->runYoloGPU(img);
        }
        else
        {
            result = net->runYolo(img);
        }
#else
        if(runGPU>0)
        {
            throw Msnhnet::Exception(1,"Msnhnet is not compiled with GPU!",__FILE__,__LINE__,__FUNCTION__);
        }
        else
        {
            result = net->runYolo(img);
        }
#endif

        if(result[0].size() > MaxBBoxNum)
        {
            throw Msnhnet::Exception(1,"BBox buffer is not enough, please change MaxBBoxNum and rebuild", __FILE__, __LINE__, __FUNCTION__);
        }

        *detectedNum = result[0].size();
        for (int i = 0; i < result[0].size(); ++i)
        {
            BBox box;
            box.x           = result[0][i].xywhBox.x;
            box.y           = result[0][i].xywhBox.y;
            box.w           = result[0][i].xywhBox.w;
            box.h           = result[0][i].xywhBox.h;
            box.conf        = result[0][i].conf;
            box.bestClsConf = result[0][i].bestClsConf;
            box.bestClsIdx  = result[0][i].bestClsIdx;
            box.angle       = result[0][i].angle;

            bboxContainer->boxes[i] = box;
        }

    }
    catch(Msnhnet::Exception ex)
    {
        std::string tmpMsg;
        tmpMsg = tmpMsg + std::string(ex.what()) + " " + ex.getErrFile() + " " + std::to_string(ex.getErrLine()) + " " + ex.getErrFun() + " \r\n";
        std::vector<char> chr{tmpMsg.data(),tmpMsg.data()+tmpMsg.length()};
        *msg = chr.data();
        return -1;
    }
    return 1;
}

int runYoloList(char **msg, unsigned char *imageData, const int width, const int height, const int channel, BBoxContainer *bboxContainer,  int *detectedNum, const int swapRGB, const int runGPU)
{
    try
    {
        if(channel != 1 && channel !=3 )
        {
            throw Msnhnet::Exception(1,"channel must be 1 or 3", __FILE__, __LINE__, __FUNCTION__);
        }

        int sizeX = 0;
        int sizeY = 0;
        Msnhnet::Point2I inputSize = net->getInputSize();
        sizeX = inputSize.x;
        sizeY = inputSize.y;

        cv::Mat inMat(height, width,  ((channel)==1)?CV_8UC1:CV_8UC3,imageData,0);

        if(swapRGB)
        {
            cv::cvtColor(inMat,inMat,cv::COLOR_RGB2BGR);
        }

        std::vector<float> img = Msnhnet::OpencvUtil::getPaddingZeroF32C3(inMat,cv::Size(sizeX,sizeY));

        std::vector<std::vector<Msnhnet::YoloBox>> result;

#ifdef USE_GPU
        if(runGPU>0)
        {
            result =  net->runYoloGPU(img);
        }
        else
        {
            result = net->runYolo(img);
        }
#else
        if(runGPU>0)
        {
            throw Msnhnet::Exception(1,"Msnhnet is not compiled with GPU!",__FILE__,__LINE__,__FUNCTION__);
        }
        else
        {
            result = net->runYolo(img);
        }
#endif

        if(result[0].size() > MaxBBoxNum)
        {
            throw Msnhnet::Exception(1,"BBox buffer is not enough, please change MaxBBoxNum and rebuild", __FILE__, __LINE__, __FUNCTION__);
        }

        *detectedNum = result[0].size();
        for (int i = 0; i < result[0].size(); ++i)
        {
            BBox box;
            box.x           = result[0][i].xywhBox.x;
            box.y           = result[0][i].xywhBox.y;
            box.w           = result[0][i].xywhBox.w;
            box.h           = result[0][i].xywhBox.h;
            box.conf        = result[0][i].conf;
            box.bestClsConf = result[0][i].bestClsConf;
            box.bestClsIdx  = result[0][i].bestClsIdx;
            box.angle       = result[0][i].angle;

            bboxContainer->boxes[i] = box;
        }

    }
    catch(Msnhnet::Exception ex)
    {
        std::string tmpMsg;
        tmpMsg = tmpMsg + std::string(ex.what()) + " " + ex.getErrFile() + " " + std::to_string(ex.getErrLine()) + " " + ex.getErrFun() + " \r\n";
        std::vector<char> chr{tmpMsg.data(),tmpMsg.data()+tmpMsg.length()};
        *msg = chr.data();
        return -1;
    }
    return 1;
}
