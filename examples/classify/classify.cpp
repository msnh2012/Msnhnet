#include <iostream>
#include "Msnhnet/net/MsnhNetBuilder.h"
#include "Msnhnet/utils/MsnhExVector.h"
#include "Msnhnet/utils/MsnhOpencvUtil.h"
#include "Msnhnet/io/MsnhIO.h"
#include "Msnhnet/config/MsnhnetCfg.h"

int main(int argc, char** argv) 
{
    if(argc != 2)
    {
        std::cout<<"\nYou need to give models folder path and the.\neg: lenet5 /your/models/folder/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }
    std::string root = argv[1];
    std::string imgPath = "../images/cat.jpg";
    try
    {
        std::vector<float> img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,cv::Size(227,227));
        std::string config_file_ = root + "/alexnet/alexnet.msnhnet";
        Msnhnet::NetBuilder  msnhNet;
        // ================================ check alexnet ================================
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/alexnet/alexnet.msnhbin");

        std::vector<float> result =  msnhNet.runClassify(img);

        int bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));

        if( (abs(Msnhnet::ExVector::max<float>(result)-10.5764f) < 0.0001) && bestIndex==331)
        {
            std::cout<<"\n===== alexnet check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== alexnet check failed ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[10.57645]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  331   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================

        // =============================== check darknet53 ==============================
        img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,cv::Size(224,224));
        config_file_ = root + "/darknet53/darknet53.msnhnet";
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/darknet53/darknet53.msnhbin");
        result =  msnhNet.runClassify(img);
        bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
        if( (abs(Msnhnet::ExVector::max<float>(result)-13.60f) < 0.01) && bestIndex==285)
        {
            std::cout<<"\n===== darknet53 check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== darknet53 check failed ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[13.60138]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  285   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================


        // =============================== check googLenet ==============================
        img = Msnhnet::OpencvUtil::getGoogLenetF32C3(imgPath,cv::Size(224,224));
        config_file_ = root + "/googLenet/googLenet.msnhnet";
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/googLenet/googLenet.msnhbin");
        result =  msnhNet.runClassify(img);
        bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
        if( (abs(Msnhnet::ExVector::max<float>(result)-6.24f) < 0.05) && bestIndex==284)
        {
            std::cout<<"\n===== googLenet check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== googLenet check failed ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[6.26569 ]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  284   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================

        // =============================== check mobilenetv2 =============================
        img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,cv::Size(224,224));
        config_file_ = root + "/mobilenetv2/mobilenetv2.msnhnet";
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/mobilenetv2/mobilenetv2.msnhbin");
        result =  msnhNet.runClassify(img);
        bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
        if( (abs(Msnhnet::ExVector::max<float>(result)-12.63f) < 1.f) && bestIndex==285) //12.63
        {
            std::cout<<"\n===== mobilenetv2 check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== mobilenetv2 check failed  ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[12.63113]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  285   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================


        // =============================== check resnet18 =============================
        img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,cv::Size(224,224));
        config_file_ = root + "/resnet18/resnet18.msnhnet";
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/resnet18/resnet18.msnhbin");
        result =  msnhNet.runClassify(img);
        bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
        if( (abs(Msnhnet::ExVector::max<float>(result)-11.10f) < 0.01f) && bestIndex==285)
        {
            std::cout<<"\n===== resnet18 check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== resnet18 check failed  ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[11.10028]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  285   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================

        // =============================== check resnet34 =============================`
        img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,cv::Size(224,224));
        config_file_ = root + "/resnet34/resnet34.msnhnet";
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/resnet34/resnet34.msnhbin");
        result =  msnhNet.runClassify(img);
        bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
        if( (abs(Msnhnet::ExVector::max<float>(result)-12.080f) < 0.01f) && bestIndex==285)
        {
            std::cout<<"\n===== resnet34 check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== resnet34 check failed  ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[12.08040]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  285   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================

        // =============================== check resnet50 =============================
        img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,cv::Size(224,224));
        config_file_ = root + "/resnet50/resnet50.msnhnet";
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/resnet50/resnet50.msnhbin");
        result =  msnhNet.runClassify(img);
        bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
        if( (abs(Msnhnet::ExVector::max<float>(result)-13.036f) < 0.001f) && bestIndex==285)
        {
            std::cout<<"\n===== resnet50 check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== resnet50 check failed  ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[13.03649]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  285   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================

        // =============================== check resnet101 =============================
        img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,cv::Size(224,224));
        config_file_ = root + "/resnet101/resnet101.msnhnet";
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/resnet101/resnet101.msnhbin");
        result =  msnhNet.runClassify(img);
        bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
        if( (abs(Msnhnet::ExVector::max<float>(result)-9.273f) < 1.f) && bestIndex==285)
        {
            std::cout<<"\n===== resnet101 check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== resnet101 check failed  ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[9.273776]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  285   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================


        // =============================== check resnet152 =============================
        img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,cv::Size(224,224));
        config_file_ = root + "/resnet152/resnet152.msnhnet";
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/resnet152/resnet152.msnhbin");
        result =  msnhNet.runClassify(img);
        bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
        if( (abs(Msnhnet::ExVector::max<float>(result)-11.60f) < 1.f) && bestIndex==285)
        {
            std::cout<<"\n===== resnet152 check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== resnet152 check failed  ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[11.60315]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  285   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================

        // =============================== check vgg16 ==================================
        img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,cv::Size(224,224));
        config_file_ = root + "/vgg16/vgg16.msnhnet";
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/vgg16/vgg16.msnhbin");
        result =  msnhNet.runClassify(img);
        bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
        if( (abs(Msnhnet::ExVector::max<float>(result)-12.1254f) < 0.001f) && bestIndex==285)
        {
            std::cout<<"\n===== vgg16 check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== vgg16 check failed  ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[12.12543]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  285   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================

        // =============================== check vgg16_bn ================================
        img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath,cv::Size(224,224));
        config_file_ = root + "/vgg16_bn/vgg16_bn.msnhnet";
        msnhNet.buildNetFromMsnhNet(config_file_);
        msnhNet.loadWeightsFromMsnhBin(root + "/vgg16_bn/vgg16_bn.msnhbin");
        result =  msnhNet.runClassify(img);
        bestIndex = static_cast<int>(Msnhnet::ExVector::maxIndex(result));
        if( (abs(Msnhnet::ExVector::max<float>(result)-9.224f) < 0.01f) && bestIndex==285)
        {
            std::cout<<"\n===== vgg16_bn check success ====="<<std::endl;
        }
        else
        {
            std::cout<<"\n===== vgg16_bn check failed  ====="<<std::endl;
        }
        std::cout<<"max   : pytorch[9.22434 ]  msnhnet: " << Msnhnet::ExVector::max<float>(result)<<std::endl;
        std::cout<<"index : pytorch[  285   ]  msnhnet: " << bestIndex<<std::endl;
        std::cout<<"time  : " << msnhNet.getInferenceTime()<<"s"<<std::endl;
        // ==============================================================================
    }
    catch(Msnhnet::Exception &ex)
    {
		std::cout << ex.what() << " { " << ex.getErrFile() << " " << ex.getErrLine() << "}";
    }
    
    return 0;
}
