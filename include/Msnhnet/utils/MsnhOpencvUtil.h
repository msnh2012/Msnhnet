#ifndef MSNHOPENCVUTIL_H
#define MSNHOPENCVUTIL_H
#ifdef USE_OPENCV

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/layers/MsnhYoloDef.h"
#include "Msnhnet/layers/MsnhYoloOutLayer.h"

namespace Msnhnet
{
class MsnhNet_API OpencvUtil
{
public:
    OpencvUtil();

    static std::vector<cv::Scalar> colorTable;

    static std::vector<float> getImgDataF32C1(const std::string &path, const cv::Size &size);
    static std::vector<float> getImgDataF32C1(cv::Mat &mat, const cv::Size &size);

    static std::vector<float> getImgDataF32C3(const std::string &path,  const cv::Size &size, const bool &halfInit = false, const bool &needShuffleRGB=false);
    static std::vector<float> getImgDataF32C3(cv::Mat &mat,  const cv::Size &size, const bool &halfInit = false, const bool &needShuffleRGB=false);

    static std::vector<float> getGoogLenetF32C3(const std::string &path,  const cv::Size &size, const bool &needShuffleRGB=false);
    static std::vector<float> getGoogLenetF32C3(cv::Mat &mat,  const cv::Size &size, const bool &needShuffleRGB=false);

    static std::vector<float> getPaddingZeroF32C3(const std::string &path,  const cv::Size &size, const bool &halfInit = false, const bool &needShuffleRGB=true);
    static std::vector<float> getPaddingZeroF32C3(cv::Mat &mat,  const cv::Size &size, const bool &halfInit = false, const bool &needShuffleRGB=true);

    static std::vector<float> getTransformedF32C3(const std::string &path, const cv::Size &size,
                                                   const cv::Scalar &mean, const cv::Scalar &std, const bool &needShuffleRGB=false);
    static std::vector<float> getTransformedF32C3(cv::Mat &mat, const cv::Size &size,
                                                   const cv::Scalar &mean, const cv::Scalar &std, const bool &needShuffleRGB=false);

    static std::vector<float> getCaffeModeF32C3(const std::string &path,  const cv::Size &size, const bool &needShuffleRGB=false);
    static std::vector<float> getCaffeModeF32C3(cv::Mat &mat,  const cv::Size &size, const bool &needShuffleRGB=false);

    static void drawYoloBox(cv::Mat &mat, std::vector<string> &labels, std::vector<std::vector<Msnhnet::YoloBox>> &boxs, const Point2I &size, const bool &noPad = false);

    static void drawSegMask(const int &channel, const int &wxh, std::vector<float> &inVal,cv::Mat &mask);
};
}
#endif
#endif 

