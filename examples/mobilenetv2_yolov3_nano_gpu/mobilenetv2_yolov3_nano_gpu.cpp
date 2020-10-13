#include <iostream>
#include "Msnhnet/Msnhnet.h"
#ifdef USE_OPENCV
void mobilenetYolov3NanoGPUOpencv(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& imgPath, const std::string& labelsPath)
{
	try
	{
		Msnhnet::NetBuilder  msnhNet;
		Msnhnet::NetBuilder::setOnlyGpu(true);
		msnhNet.buildNetFromMsnhNet(msnhnetPath);
		std::cout << msnhNet.getLayerDetail();
		msnhNet.loadWeightsFromMsnhBin(msnhbinPath);
		std::vector<std::string> labels;
		Msnhnet::IO::readVectorStr(labels, labelsPath.data(), "\n");
		Msnhnet::Point2I inSize = msnhNet.getInputSize();

		std::vector<float> img;
		std::vector<std::vector<Msnhnet::YoloBox>> result;

		img = Msnhnet::OpencvUtil::getPaddingZeroF32C3(imgPath, cv::Size(inSize.x, inSize.y));
		for (size_t i = 0; i < 10; i++)
		{
			auto st = Msnhnet::TimeUtil::startRecord();
			result = msnhNet.runYoloGPU(img);
			std::cout << "time  : " << Msnhnet::TimeUtil::getElapsedTime(st) << "ms" << std::endl << std::flush;
		}

		cv::Mat org = cv::imread(imgPath);
		Msnhnet::OpencvUtil::drawYoloBox(org, labels, result, inSize);
		cv::imshow("test", org);
		cv::waitKey();
	}
	catch (Msnhnet::Exception ex)
	{
		std::cout << ex.what() << " " << ex.getErrFile() << " " << ex.getErrLine() << " " << ex.getErrFun() << std::endl;
	}
}
#endif

void mobilenetYolov3NanoGPUMsnhCV(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& imgPath, const std::string& labelsPath)
{
	try
	{
		Msnhnet::NetBuilder  msnhNet;
		Msnhnet::NetBuilder::setOnlyGpu(true);
		msnhNet.buildNetFromMsnhNet(msnhnetPath);
		std::cout << msnhNet.getLayerDetail();
		msnhNet.loadWeightsFromMsnhBin(msnhbinPath);
		std::vector<std::string> labels;
		Msnhnet::IO::readVectorStr(labels, labelsPath.data(), "\n");
		Msnhnet::Point2I inSize = msnhNet.getInputSize();

		std::vector<float> img;
		std::vector<std::vector<Msnhnet::YoloBox>> result;

		img = Msnhnet::CVUtil::getPaddingZeroF32C3(imgPath, { inSize.x,inSize.y });
		for (size_t i = 0; i < 10; i++)
		{
			auto st = Msnhnet::TimeUtil::startRecord();
			result = msnhNet.runYoloGPU(img);
			std::cout << "time  : " << Msnhnet::TimeUtil::getElapsedTime(st) << "ms" << std::endl << std::flush;
		}

		Msnhnet::Mat org(imgPath);
		Msnhnet::CVUtil::drawYoloBox(org, labels, result, inSize);
		org.saveImage("mobilenet_yolov3_nano_gpu.jpg");
		system("mobilenet_yolov3_nano_gpu.jpg");
	}
	catch (Msnhnet::Exception ex)
	{
		std::cout << ex.what() << " " << ex.getErrFile() << " " << ex.getErrLine() << " " << ex.getErrFun() << std::endl;
	}
}

int main(int argc, char** argv) 
{
    if(argc != 2)
    {
        std::cout<<"\nYou need to give models dir path.\neg: yolov3 /your/models/dir/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }

    std::string msnhnetPath = std::string(argv[1]) + "/mobilenetv2_yolov3_nano/mobilenetv2_yolov3_nano.msnhnet";
    std::string msnhbinPath = std::string(argv[1]) + "/mobilenetv2_yolov3_nano/mobilenetv2_yolov3_nano.msnhbin";
    std::string labelsPath  = "../labels/coco.names";
    std::string imgPath = "../images/dog.jpg";
#ifdef USE_OPENCV
	mobilenetYolov3NanoGPUOpencv(msnhnetPath, msnhbinPath, imgPath, labelsPath);
#else
	mobilenetYolov3NanoGPUMsnhCV(msnhnetPath, msnhbinPath, imgPath, labelsPath);
#endif
    return 0;
}
