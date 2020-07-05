#ifndef MSNHOPENCVUTIL_H
#define MSNHOPENCVUTIL_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "MsnhException.h"
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/layers/MsnhYolov3Def.h"
#include "Msnhnet/layers/MsnhYolov3OutLayer.h"

namespace Msnhnet
{
class MsnhNet_API OpencvUtil
{
public:
    OpencvUtil();

   static std::vector<cv::Scalar> colorTable;

   static std::vector<float> getImgDataF32C1(const std::string &path, const cv::Size &size);
    static std::vector<float> getImgDataF32C1(cv::Mat &mat, const cv::Size &size);

   static std::vector<float> getImgDataF32C3(const std::string &path,  const cv::Size &size);
    static std::vector<float> getImgDataF32C3(cv::Mat &mat,  const cv::Size &size);

   static std::vector<float> getGoogLenetF32C3(const std::string &path,  const cv::Size &size);
    static std::vector<float> getGoogLenetF32C3(cv::Mat &mat,  const cv::Size &size);

   static std::vector<float> getPaddingZeroF32C3(const std::string &path,  const cv::Size &size);
    static std::vector<float> getPaddingZeroF32C3(cv::Mat &mat,  const cv::Size &size);

   static void drawYolov3Box(cv::Mat &mat, std::vector<string> &labels, std::vector<std::vector<Msnhnet::Yolov3Box>> &boxs);
};
}

#endif 

