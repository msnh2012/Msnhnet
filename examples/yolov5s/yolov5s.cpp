#include <iostream>
#include "Msnhnet/Msnhnet.h"


#ifdef USE_OPENCV
void yolov5sOpencv(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& imgPath,  const std::string& labelsPath)
{
    try
    {
        Msnhnet::NetBuilder  msnhNet;
        msnhNet.buildNetFromMsnhNet(msnhnetPath);
        std::cout<<msnhNet.getLayerDetail();
        msnhNet.loadWeightsFromMsnhBin(msnhbinPath);
        std::vector<std::string> labels ;
        Msnhnet::IO::readVectorStr(labels, labelsPath.data(), "\n");
        Msnhnet::Point2I inSize = msnhNet.getInputSize();
        std::vector<float> img = Msnhnet::OpencvUtil::getPaddingZeroF32C3(imgPath, cv::Size(inSize.x,inSize.y));
        cv::Mat org = cv::imread(imgPath);
        std::vector<std::vector<Msnhnet::YoloBox>> result = msnhNet.runYolo(img);
        Msnhnet::OpencvUtil::drawYoloBox(org,labels,result,inSize);
        std::cout<<msnhNet.getTimeDetail()<<std::endl<<std::flush;
        cv::imshow("test",org);
        cv::waitKey();
    }
    catch (Msnhnet::Exception ex)
    {
        std::cout<<ex.what()<<" "<<ex.getErrFile() << " " <<ex.getErrLine()<< " "<<ex.getErrFun()<<std::endl;
    }
}
#endif

void yolov5sMsnhCV(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& imgPath,  const std::string& labelsPath)
{
    try
    {
        Msnhnet::NetBuilder  msnhNet;
        msnhNet.buildNetFromMsnhNet(msnhnetPath);
        std::cout<<msnhNet.getLayerDetail();
        msnhNet.loadWeightsFromMsnhBin(msnhbinPath);
        std::vector<std::string> labels ;
        Msnhnet::IO::readVectorStr(labels, labelsPath.data(), "\n");
        Msnhnet::Point2I inSize = msnhNet.getInputSize();
        std::vector<float> img = Msnhnet::CVUtil::getPaddingZeroF32C3(imgPath, {inSize.x,inSize.y});
        Msnhnet::Mat org(imgPath);
        std::vector<std::vector<Msnhnet::YoloBox>> result = msnhNet.runYolo(img);
        Msnhnet::CVUtil::drawYoloBox(org,labels,result,inSize);
        std::cout<<msnhNet.getTimeDetail()<<std::endl<<std::flush;
        org.saveImage("yolov5s.jpg");
        system("yolov5s.jpg");
    }
    catch (Msnhnet::Exception ex)
    {
        std::cout<<ex.what()<<" "<<ex.getErrFile() << " " <<ex.getErrLine()<< " "<<ex.getErrFun()<<std::endl;
    }
}

int main(int argc, char** argv) 
{
    if(argc != 2)
    {
        std::cout<<"\nYou need to give models dir path.\neg: yolov5s /your/models/dir/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }

    std::string msnhnetPath = std::string(argv[1]) + "/yolov5s/yolov5s.msnhnet";
    std::string msnhbinPath = std::string(argv[1]) + "/yolov5s/yolov5s.msnhbin";
    std::string labelsPath  = "../labels/coco.names";
    std::string imgPath = "../images/dog.jpg";
#ifdef USE_OPENCV
    yolov5sOpencv(msnhnetPath, msnhbinPath, imgPath,labelsPath);
#else
    yolov5sMsnhCV(msnhnetPath, msnhbinPath, imgPath, labelsPath);
#endif
    getchar();

    return 0;
}
