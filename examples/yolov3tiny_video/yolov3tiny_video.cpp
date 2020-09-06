#include <iostream>
#include "Msnhnet/Msnhnet.h"

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
            std::vector<std::vector<Msnhnet::Yolov3Box>> result = msnhNet.runYolov3(img);
            Msnhnet::OpencvUtil::drawYolov3Box(org,labels,result,inSize);
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
    return 0;
}
