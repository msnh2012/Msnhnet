#include <iostream>
#include "Msnhnet/net/MsnhNetBuilder.h"
#include "Msnhnet/io/MsnhIO.h"
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/utils/MsnhOpencvUtil.h"

int main(int argc, char** argv) 
{
    if(argc != 2)
    {
        std::cout<<"\nYou need to give models dir path.\neg: fcns /your/models/dir/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }

    std::string msnhnetPath = std::string(argv[1]) + "/fcns/fcns.msnhnet";
    std::string msnhbinPath = std::string(argv[1]) + "/fcns/fcns.msnhbin";
    std::string imgPath = "../images/fcns.jpg";
    try
    {
        Msnhnet::NetBuilder  msnhNet;
        msnhNet.buildNetFromMsnhNet(msnhnetPath);
        std::cout<<msnhNet.getLayerDetail();
        msnhNet.loadWeightsFromMsnhBin(msnhbinPath);

        int netX  = msnhNet.getInputSize().x;
        int netY  = msnhNet.getInputSize().y;

        std::vector<float> img = Msnhnet::OpencvUtil::getTransformedF32C3(imgPath,{netX,netY},cv::Scalar(0.485, 0.456, 0.406),cv::Scalar(0.229, 0.224, 0.225));
        std::vector<float> result =  msnhNet.runClassify(img);
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
        std::cout<<msnhNet.getTimeDetail()<<std::endl;
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
