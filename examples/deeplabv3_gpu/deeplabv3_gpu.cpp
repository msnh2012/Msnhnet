#include <iostream>
#include "Msnhnet/Msnhnet.h"

int main(int argc, char** argv) 
{
    if(argc != 2)
    {
        std::cout<<"\nYou need to give models dir path.\neg: deeplabv3_gpu /your/models/dir/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }

    std::string msnhnetPath = std::string(argv[1]) + "/deeplabv3/deeplabv3.msnhnet";
    std::string msnhbinPath = std::string(argv[1]) + "/deeplabv3/deeplabv3.msnhbin";
    std::string imgPath = "../images/deeplabv3.jpg";
    try
    {
        Msnhnet::NetBuilder  msnhNet;
        Msnhnet::NetBuilder::setOnlyGpu(true);

        msnhNet.buildNetFromMsnhNet(msnhnetPath);
        std::cout<<msnhNet.getLayerDetail();
        msnhNet.loadWeightsFromMsnhBin(msnhbinPath);

        int netX  = msnhNet.getInputSize().x;
        int netY  = msnhNet.getInputSize().y;

        std::vector<float> img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,{netX,netY});
        std::vector<float> result;

        for (size_t i = 0; i < 10; i++)
        {
			auto st = Msnhnet::TimeUtil::startRecord();
            result = msnhNet.runClassifyGPU(img);
            std::cout<<"time  : " << Msnhnet::TimeUtil::getElapsedTime(st) <<"ms"<<std::endl<<std::flush;
        }

        cv::Mat mat = cv::imread(imgPath);
        cv::resize(mat, mat,{netX,netY});

        cv::imshow("org",mat);

        cv::Mat mask(netX,netY,CV_8UC3,cv::Scalar(0,0,0));
        int outChannel = msnhNet.getLastLayerOutChannel();
        int outSize     = msnhNet.getLastLayerOutHeight()*msnhNet.getLastLayerOutWidth();
        Msnhnet::OpencvUtil::drawSegMask(outChannel, outSize, result, mask);

        mat = mat + mask;
        cv::imshow("mask",mask);
        cv::imshow("get",mat);
        cv::waitKey(0);

    }
    catch (Msnhnet::Exception ex)
    {
        std::cout<<ex.what()<<" "<<ex.getErrFile() << " " <<ex.getErrLine()<< " "<<ex.getErrFun()<<std::endl;
    }
    return 0;
}
