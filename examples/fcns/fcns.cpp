#include <iostream>
#include "Msnhnet/Msnhnet.h"

#ifdef USE_OPENCV
void fcnsOpencv(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& imgPath)
{
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
        std::cout<<ex.what()<<" "<<ex.getErrFile() << " " <<ex.getErrLine()<< " "<<ex.getErrFun()<<std::endl;
    }
}
#endif


void fcnsMsnhCV(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& imgPath)
{
    try
    {
        Msnhnet::NetBuilder  msnhNet;
        msnhNet.buildNetFromMsnhNet(msnhnetPath);
        std::cout<<msnhNet.getLayerDetail();
        msnhNet.loadWeightsFromMsnhBin(msnhbinPath);

        int netX  = msnhNet.getInputSize().x;
        int netY  = msnhNet.getInputSize().y;

        std::vector<float> img = Msnhnet::CVUtil::getTransformedF32C3(imgPath,{netX,netY},{0.485, 0.456, 0.406},{0.229, 0.224, 0.225});
        std::vector<float> result =  msnhNet.runClassify(img);
        Msnhnet::Mat mat(imgPath);

        Msnhnet::Mat mask(netX,netY, Msnhnet::MatType::MAT_RGB_U8);

        for (int i = 0; i < result.size()/2; ++i)
        {
            if(result[i] < result[i+msnhNet.getInputSize().x*msnhNet.getInputSize().y])
            {
                mask.getData().u8[i*3+0] += 120;
            }
        }

        Msnhnet::MatOp::resize(mask,mask,{mat.getWidth(),mat.getHeight()});
        std::cout<<msnhNet.getTimeDetail()<<std::endl;
        mat = mat + mask;

        #ifdef USE_MSNHCV_GUI
        Msnhnet::Gui::imShow("fcns",mat);
        Msnhnet::Gui::imShow("fcns_mask",mask);
        #else
        mat.saveImage("fcns.jpg");
		mask.saveImage("fcns_mask.jpg");
        #endif
        
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
        std::cout<<"\nYou need to give models dir path.\neg: fcns /your/models/dir/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }

    std::string msnhnetPath = std::string(argv[1]) + "/fcns/fcns.msnhnet";
    std::string msnhbinPath = std::string(argv[1]) + "/fcns/fcns.msnhbin";
    std::string imgPath = "../images/fcns.jpg";
#ifdef USE_OPENCV
    fcnsOpencv(msnhnetPath, msnhbinPath, imgPath);
#else
    fcnsMsnhCV(msnhnetPath, msnhbinPath, imgPath);
#endif
    return 0;
}
