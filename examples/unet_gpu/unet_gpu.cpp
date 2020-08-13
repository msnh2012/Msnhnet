#include <iostream>
#include "Msnhnet/Msnhnet.h"

int main(int argc, char** argv) 
{
    if(argc != 2)
    {
        std::cout<<"\nYou need to give models dir path.\neg: unet /your/models/dir/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }

    std::string msnhnetPath = std::string(argv[1]) + "/unet/unet.msnhnet";
    std::string msnhbinPath = std::string(argv[1]) + "/unet/unet.msnhbin";
    std::string imgPath = "../images/unet.jpg";
    try
    {
        Msnhnet::NetBuilder  msnhNet;
        msnhNet.buildNetFromMsnhNet(msnhnetPath);
        std::cout<<msnhNet.getLayerDetail();
        msnhNet.loadWeightsFromMsnhBin(msnhbinPath);

        int netX  = msnhNet.getInputSize().x;
        int netY  = msnhNet.getInputSize().y;

        std::vector<float> img;
        std::vector<float> result;
        img  = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,{netX,netY});
        
        for (size_t i = 0; i < 10; i++)
        {
			auto st = Msnhnet::TimeUtil::startRecord();
            result =  msnhNet.runClassifyGPU(img);
            std::cout<<"time  : " << Msnhnet::TimeUtil::getElapsedTime(st) <<"ms"<<std::endl;
        }
        
        cv::Mat mat = cv::imread(imgPath);

        cv::imshow("org",mat);

        cv::Mat mask(netX,netY,CV_8UC3,cv::Scalar(0,0,0));

        for (int i = 0; i < result.size()/2; ++i)
        {
            if(result[i] < result[i+msnhNet.getInputSize().x*msnhNet.getInputSize().y])
            {
                mask.data[i*3+2] += 120;
            }
        }

        cv::medianBlur(mask,mask,11);
        cv::resize(mask,mask,{mat.rows,mat.cols});
        mat = mat + mask;
        cv::imshow("get",mat);
        cv::waitKey();

    }
    catch (Msnhnet::Exception ex)
    {
        std::cout<<ex.what()<<" "<<ex.getErrFile() << " " <<ex.getErrLine()<< " "<<ex.getErrFun()<<std::endl;
    }
    return 0;
}
