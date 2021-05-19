#include <iostream>
#include "Msnhnet/Msnhnet.h"

#ifdef USE_OPENCV
void unetOpencv(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& imgPath)
{
	try
	{
		Msnhnet::NetBuilder  msnhNet;
		Msnhnet::NetBuilder::setOnlyGpu(true);
		Msnhnet::NetBuilder::setUseFp16(true);
		msnhNet.buildNetFromMsnhNet(msnhnetPath);
		std::cout << msnhNet.getLayerDetail();
		msnhNet.loadWeightsFromMsnhBin(msnhbinPath);

		int netX = msnhNet.getInputSize().x;
		int netY = msnhNet.getInputSize().y;

		std::vector<float> img;
		std::vector<float> result;
		img = Msnhnet::OpencvUtil::getImgDataF32C3(imgPath, { netX,netY });

		for (size_t i = 0; i < 10; i++)
		{
			auto st = Msnhnet::TimeUtil::startRecord();
			result = msnhNet.runClassifyGPU(img);
			std::cout << "time  : " << Msnhnet::TimeUtil::getElapsedTime(st) << "ms" << std::endl;
		}

		cv::Mat mat = cv::imread(imgPath);

		cv::imshow("org", mat);

		cv::Mat mask(netX, netY, CV_8UC3, cv::Scalar(0, 0, 0));

		for (int i = 0; i < result.size() / 2; ++i)
		{
			if (result[i] < result[i + msnhNet.getInputSize().x*msnhNet.getInputSize().y])
			{
				mask.data[i * 3 + 2] += 120;
			}
		}

		cv::medianBlur(mask, mask, 11);
		cv::resize(mask, mask, { mat.rows,mat.cols });
		mat = mat + mask;
		cv::imshow("get", mat);
		cv::waitKey();

	}
	catch (Msnhnet::Exception ex)
	{
		std::cout << ex.what() << " " << ex.getErrFile() << " " << ex.getErrLine() << " " << ex.getErrFun() << std::endl;
	}
}
#endif

void unetMsnhCV(const std::string& msnhnetPath, const std::string& msnhbinPath, const std::string& imgPath)
{
	try
	{
		Msnhnet::NetBuilder  msnhNet;
		Msnhnet::NetBuilder::setOnlyGpu(true);
		Msnhnet::NetBuilder::setUseFp16(true);
		msnhNet.buildNetFromMsnhNet(msnhnetPath);
		std::cout << msnhNet.getLayerDetail();
		msnhNet.loadWeightsFromMsnhBin(msnhbinPath);

		int netX = msnhNet.getInputSize().x;
		int netY = msnhNet.getInputSize().y;

		std::vector<float> img;
		std::vector<float> result;
		img = Msnhnet::CVUtil::getImgDataF32C3(imgPath, { netX,netY });

		for (size_t i = 0; i < 10; i++)
		{
			auto st = Msnhnet::TimeUtil::startRecord();
			result = msnhNet.runClassifyGPU(img);
			std::cout << "time  : " << Msnhnet::TimeUtil::getElapsedTime(st) << "ms" << std::endl;
		}

		Msnhnet::Mat mat(imgPath);

		Msnhnet::Mat mask(netX, netY, Msnhnet::MatType::MAT_RGB_U8);

		for (int i = 0; i < result.size() / 2; ++i)
		{
			if (result[i] < result[i + msnhNet.getInputSize().x*msnhNet.getInputSize().y])
			{
				mask.getData().u8[i * 3 + 0] += 120;
			}
		}

		Msnhnet::MatOp::resize(mask, mask, { mat.getWidth(),mat.getHeight() });
		
		if (mat.getChannel() == 1)
			Msnhnet::MatOp::cvtColor(mat, mat, Msnhnet::CVT_GRAY2RGB);

		mat = mat + mask;

		#ifdef USE_MSNHCV_GUI
        Msnhnet::Gui::imShow("unet_gpu_fp16",mat);
		Msnhnet::Gui::wait();
        #else
        mat.saveImage("unet_gpu_fp16.jpg");
        #endif
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
        std::cout<<"\nYou need to give models dir path.\neg: unet /your/models/dir/path/ \n\nModels folder must be like this:\nmodels\n  |-Lenet5\n    |-Lenet5.msnhnet\n    |-Lenet5.msnhbin";
        getchar();
        return 0;
    }

    std::string msnhnetPath = std::string(argv[1]) + "/unet/unet.msnhnet";
    std::string msnhbinPath = std::string(argv[1]) + "/unet/unet.msnhbin";
    std::string imgPath = "../images/unet.jpg";
#ifdef USE_OPENCV
	unetOpencv(msnhnetPath, msnhbinPath, imgPath);
#else
	unetMsnhCV(msnhnetPath, msnhbinPath, imgPath);
#endif
    return 0;
}
