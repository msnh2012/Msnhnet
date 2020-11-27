#ifndef MSNHCVUTIL_H
#define MSNHCVUTIL_H

#include <vector>
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/layers/MsnhYoloDef.h"
#include "Msnhnet/layers/MsnhYoloOutLayer.h"
#include "Msnhnet/cv/MsnhCV.h"

namespace Msnhnet
{
class MsnhNet_API CVUtil
{
public:
    CVUtil();

    static std::vector<Vec3U8> colorTable;

    static std::vector<float> getImgDataF32C1(const std::string &path, const Vec2I32 &size);
    static std::vector<float> getImgDataF32C1(Mat &mat, const Vec2I32 &size);

    static std::vector<float> getImgDataF32C3(const std::string &path,  const Vec2I32 &size, const bool &halfInit = false, const bool &needShuffleRGB=true);
    static std::vector<float> getImgDataF32C3(Mat &mat,  const Vec2I32 &size, const bool &halfInit = false, const bool &needShuffleRGB=true);

    static std::vector<float> getGoogLenetF32C3(const std::string &path,  const Vec2I32 &size, const bool &needShuffleRGB=true);
    static std::vector<float> getGoogLenetF32C3(Mat &mat,  const Vec2I32 &size, const bool &needShuffleRGB=true);

    static std::vector<float> getPaddingZeroF32C3(const std::string &path,  const Vec2I32 &size, const float &halfInit=false, const bool &needShuffleRGB=false);
    static std::vector<float> getPaddingZeroF32C3(Mat &mat,  const Vec2I32 &size, const float &halfInit=false, const bool &needShuffleRGB=false);

    static std::vector<float> getTransformedF32C3(const std::string &path, const Vec2I32 &size,
                                                   const Vec3F32 &mean, const Vec3F32 &std, const bool &needShuffleRGB=true);
    static std::vector<float> getTransformedF32C3(Mat &mat, const Vec2I32 &size,
                                                   const Vec3F32 &mean, const Vec3F32 &std, const bool &needShuffleRGB=true);

    static std::vector<float> getCaffeModeF32C3(const std::string &path,  const Vec2I32 &size, const bool &needShuffleRGB=true);
    static std::vector<float> getCaffeModeF32C3(Mat &mat,  const Vec2I32 &size, const bool &needShuffleRGB=true);

    static std::vector<Vec2I32> drawYoloBox(Mat &mat, std::vector<string> &labels, std::vector<std::vector<Msnhnet::YoloBox>> &boxs, const Point2I &size, const bool &noPad = false);

    static void drawSegMask(const int &channel, const int &wxh, std::vector<float> &inVal,Mat &mask);
};
}

#endif 

