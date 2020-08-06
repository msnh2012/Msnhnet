#include <iostream>
#include "Msnhnet/net/MsnhNetBuilder.h"
#include "Msnhnet/io/MsnhIO.h"
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/utils/MsnhOpencvUtil.h"

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
    std::string imgPath = std::string(argv[1]) + "/unet/unet.jpg";
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

        for(int i = 0;i<10;i++)
        {
            Msnhnet::TimeUtil::startRecord();
            img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,{netX,netY});
            result =  msnhNet.runClassifyGPU(img);
            std::cout<<"time  : " << Msnhnet::TimeUtil::getElapsedTime() <<"ms"<<std::endl<<std::flush;
        }

        cv::Mat mat = cv::imread(imgPath);

        cv::imshow("org",mat);

        cv::Mat mask(netX,netY,CV_8UC3,cv::Scalar(0,0,0));

        for (int i = 0; i < result.size()/2; ++i) // 2分类，多分类需修改。
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
        std::cout<<ex.what()<<std::endl;
    }
    return 0;
}
