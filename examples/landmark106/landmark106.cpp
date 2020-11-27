#include <iostream>
#include "Msnhnet/Msnhnet.h"

#ifdef USE_OPENCV
void landmark106Opencv(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& imgPath)
{
    try
    {
        Msnhnet::NetBuilder  msnhNet;
        msnhNet.buildNetFromMsnhNet(msnhnetPath);
        std::cout<<msnhNet.getLayerDetail();
        msnhNet.loadWeightsFromMsnhBin(msnhbinPath);
        Msnhnet::Point2I inSize = msnhNet.getInputSize();

        cv::Mat mat = cv::imread(imgPath);
        cv::Mat org = mat;

        if(mat.channels()==4)
        {
            cv::cvtColor(mat,mat,cv::COLOR_RGBA2RGB);
        }

        int w = org.cols;
        int h = org.rows;

        std::vector<float> img = Msnhnet::OpencvUtil::getImgDataF32C3(mat,{inSize.x,inSize.y},true,false);
        std::vector<float> result ;

        auto st = std::chrono::high_resolution_clock::now();
        result = msnhNet.runClassify(img);
        auto so = std::chrono::high_resolution_clock::now();
        std::cout<<std::chrono::duration <double,std::milli> (so-st).count()<< "---------------------------------------" << std::endl;

        for (int i = 0; i < result.size()/2; ++i)
        {
            cv::Point vec;
            vec.x = (int)(result[i*2]*w);
            vec.y = (int)(result[i*2+1]*h);
            cv::circle(org,vec,2,{0,0,255});
        }

        cv::imshow("landmark106",org);
        cv::waitKey();
    }
    catch (Msnhnet::Exception ex)
    {
        std::cout<<ex.what()<<" "<<ex.getErrFile() << " " <<ex.getErrLine()<< " "<<ex.getErrFun()<<std::endl;
    }
}
#endif

void landmark106MsnhCV(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& imgPath)
{
    try
    {
        Msnhnet::NetBuilder  msnhNet;
        msnhNet.buildNetFromMsnhNet(msnhnetPath);
        std::cout<<msnhNet.getLayerDetail();
        msnhNet.loadWeightsFromMsnhBin(msnhbinPath);
        Msnhnet::Point2I inSize = msnhNet.getInputSize();

        Msnhnet::Mat mat(imgPath);
        Msnhnet::Mat org = mat;

        if(mat.getChannel()==4)
        {
            Msnhnet::MatOp::cvtColor(mat,mat,Msnhnet::CVT_RGBA2RGB);
        }

        int w = org.getWidth();
        int h = org.getHeight();

        std::vector<float> img = Msnhnet::CVUtil::getImgDataF32C3(mat,{inSize.x,inSize.y},true,true);
        std::vector<float> result ;

        auto st = std::chrono::high_resolution_clock::now();
        result = msnhNet.runClassify(img);
        auto so = std::chrono::high_resolution_clock::now();
        std::cout<<std::chrono::duration <double,std::milli> (so-st).count()<< "---------------------------------------" << std::endl;

        for (int i = 0; i < result.size()/2; ++i)
        {
            Msnhnet::Vec2I32 vec;
            vec.x1 = result[i*2]*w;
            vec.x2 = result[i*2+1]*h;
            Msnhnet::Draw::fillEllipse(org,vec,2,2,{255,0,0});
        }
#ifdef USE_MSNHCV_GUI
        Msnhnet::Gui::imShow("landmark106",org);
        Msnhnet::Gui::wait();
#else
        org.saveImage("landmark106.jpg");
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
        std::cout<<"\nYou need to give models dir path.\neg: landmark106 /your/models/dir/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }

    std::string msnhnetPath = std::string(argv[1]) + "/landmark106/landmark106.msnhnet";
    std::string msnhbinPath = std::string(argv[1]) + "/landmark106/landmark106.msnhbin";
    std::string imgPath = "../images/landmark106.bmp";
#ifdef USE_OPENCV
    landmark106Opencv(msnhnetPath, msnhbinPath, imgPath);
#else
    landmark106MsnhCV(msnhnetPath, msnhbinPath, imgPath);
#endif
    return 0;
}
