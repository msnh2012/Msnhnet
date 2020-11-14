#include <iostream>
#include "Msnhnet/Msnhnet.h"

#ifdef USE_OPENCV
void yolov3tinyOpencvVideo(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& labelsPath)
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

        cv::VideoCapture cap(0);
        cv::Mat mat;
        if(!cap.isOpened())
        {
            std::cout<< "cam err" <<std::endl;
        }

        while (1)
        {
            cap >> mat;
            cv::Mat org = mat.clone();
            std::vector<float> img = Msnhnet::OpencvUtil::getPaddingZeroF32C3(mat, cv::Size(inSize.x,inSize.y));
            std::vector<std::vector<Msnhnet::YoloBox>> result = msnhNet.runYolo(img);
            Msnhnet::OpencvUtil::drawYoloBox(org,labels,result,inSize);
            std::cout<<msnhNet.getInferenceTime()<<std::endl;
            cv::imshow("test",org);
            if(cv::waitKey(20) == 27)
            {
                break;
            }

        }
    }
    catch (Msnhnet::Exception ex)
    {
        std::cout<<ex.what()<<" "<<ex.getErrFile() << " " <<ex.getErrLine()<< " "<<ex.getErrFun()<<std::endl;
    }
}
#endif

#ifdef USE_MSNHCV_GUI
void yolov3tinyMsnhCVVideo(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& labelsPath)
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

        Msnhnet::VideoCapture cap;
        Msnhnet::Mat mat;

        if(!cap.openCamera(0,640,480))
        {
            std::cout<< "cam err" <<std::endl;
        }

        while (1)
        {
            cap.getMat(mat);
            Msnhnet::Mat newMat = mat;
            std::vector<float> img = Msnhnet::CVUtil::getPaddingZeroF32C3(newMat, {inSize.x,inSize.y});
            std::vector<std::vector<Msnhnet::YoloBox>> result = msnhNet.runYolo(img);
            Msnhnet::CVUtil::drawYoloBox(mat,labels,result,inSize);
            std::cout<<msnhNet.getInferenceTime()<<std::endl;
            Msnhnet::Gui::imShow("test",mat);
            if (Msnhnet::Gui::waitEnterKey())
            {
                std::cout << "exit....." << std::endl;
                return;
            }
        }
    }
    catch (Msnhnet::Exception ex)
    {
        std::cout<<ex.what()<<" "<<ex.getErrFile() << " " <<ex.getErrLine()<< " "<<ex.getErrFun()<<std::endl;
    }
}
#endif

int main(int argc, char** argv) 
{
    if(argc != 2)
    {
        std::cout<<"\nYou need to give models dir path.\neg: lenet5 /your/models/dir/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }

    std::string msnhnetPath = std::string(argv[1]) + "/yolov3_tiny/yolov3_tiny.msnhnet";
    std::string msnhbinPath = std::string(argv[1]) + "/yolov3_tiny/yolov3_tiny.msnhbin";
    std::string labelsPath  = "../labels/coco.names";

#ifdef USE_OPENCV
    yolov3tinyOpencvVideo(msnhnetPath, msnhbinPath, labelsPath);
#elif USE_MSNHCV_GUI
    yolov3tinyMsnhCVVideo(msnhnetPath, msnhbinPath, labelsPath);
#endif
    return 0;
}
