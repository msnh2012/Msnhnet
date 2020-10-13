
#include "Msnhnet/utils/MsnhCVUtil.h"

namespace Msnhnet
{

std::vector<Vec3U8> CVUtil::colorTable =  {
    Vec3U8(200, 0   ,0  ), Vec3U8(  0, 200, 0  ), Vec3U8(  0,  0 , 200),
    Vec3U8(200, 255 ,0  ), Vec3U8(  0, 200, 255), Vec3U8(255,  0 , 200),
    Vec3U8(200, 0   ,50 ), Vec3U8(  0, 200, 50 ), Vec3U8( 50,  50, 200),
    Vec3U8(200, 255 ,50 ), Vec3U8( 50, 200, 255), Vec3U8(255,  50, 200),
    Vec3U8(200, 0   ,100), Vec3U8(  0, 200, 100), Vec3U8( 50, 100, 200),
    Vec3U8(200, 255 ,100), Vec3U8(100, 200, 255), Vec3U8(255, 100, 200),
    Vec3U8(200, 0   ,150), Vec3U8(  0, 200, 150), Vec3U8( 50, 150, 200),
    Vec3U8(200, 255 ,150), Vec3U8(150, 200, 255), Vec3U8(255, 150, 200),
    Vec3U8(200, 0   ,200), Vec3U8(  0, 200, 200), Vec3U8( 50, 200, 200),
    Vec3U8(200, 255 ,200), Vec3U8(200, 200, 255), Vec3U8(255, 200, 200),
    Vec3U8(150, 0   ,0  ), Vec3U8(  0, 150, 0  ), Vec3U8(  0,  0 , 150),
    Vec3U8(150, 255 ,0  ), Vec3U8(  0, 150, 255), Vec3U8(255,  0 , 150),
    Vec3U8(150, 0   ,50 ), Vec3U8(  0, 150, 50 ), Vec3U8( 50,  50, 150),
    Vec3U8(150, 255 ,50 ), Vec3U8( 50, 150, 255), Vec3U8(255,  50, 150),
    Vec3U8(150, 0   ,100), Vec3U8(  0, 150, 100), Vec3U8( 50, 100, 150),
    Vec3U8(150, 255 ,100), Vec3U8(100, 150, 255), Vec3U8(255, 100, 150),
    Vec3U8(150, 0   ,150), Vec3U8(  0, 150, 150), Vec3U8( 50, 150, 150),
    Vec3U8(150, 255 ,150), Vec3U8(150, 150, 255), Vec3U8(255, 150, 150),
    Vec3U8(150, 0   ,200), Vec3U8(  0, 150, 200), Vec3U8( 50, 200, 150),
    Vec3U8(150, 255 ,200), Vec3U8(200, 150, 255), Vec3U8(255, 200, 150),
    Vec3U8(255, 0   ,0  ), Vec3U8(  0, 255, 0  ), Vec3U8(  0,  0 , 255),
    Vec3U8(255, 255 ,0  ), Vec3U8(  0, 255, 255), Vec3U8(255,  0 , 255),
    Vec3U8(255, 0   ,50 ), Vec3U8(  0, 255, 50 ), Vec3U8( 50,  50, 255),
    Vec3U8(255, 255 ,50 ), Vec3U8( 50, 255, 255), Vec3U8(255,  50, 255),
    Vec3U8(255, 0   ,100), Vec3U8(  0, 255, 100), Vec3U8( 50, 100, 255),
    Vec3U8(255, 255 ,100), Vec3U8(100, 255, 255), Vec3U8(255, 100, 255),
    Vec3U8(255, 0   ,150), Vec3U8(  0, 255, 150), Vec3U8( 50, 150, 255),
    Vec3U8(255, 255 ,150), Vec3U8(150, 255, 255), Vec3U8(255, 150, 255),
    Vec3U8(255, 0   ,200), Vec3U8(  0, 255, 200), Vec3U8( 50, 200, 255),
    Vec3U8(255, 255 ,200), Vec3U8(200, 255, 255), Vec3U8(255, 200, 255),
};

std::vector<float> CVUtil::getImgDataF32C1(const std::string &path, const Vec2I32 &size)
{
    Mat mat;
    mat.readImage(path);
    return getImgDataF32C1(mat,size);
}

std::vector<float> CVUtil::getImgDataF32C1(Mat &mat, const Vec2I32 &size)
{
    if(mat.isEmpty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    int width = mat.getWidth();
    int height = mat.getHeight();

    std::vector<float> imgs(static_cast<size_t>(width*height));

    MatOp::resize(mat, mat, size);

    if(mat.getChannel() == 3)
    {
        MatOp::cvtColor(mat,mat,CVT_RGB2GRAY);
    }
    else if(mat.getChannel() == 4)
    {
        MatOp::cvtColor(mat,mat,CVT_RGBA2GRAY);
    }

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            imgs[static_cast<size_t>(y*width + x)] = mat.getData().u8[y*width + x ] / 255.0f;
        }
    }

    mat.release();
    return imgs;
}

std::vector<float> CVUtil::getImgDataF32C3(const std::string &path, const Vec2I32 &size, const bool &needShuffleRGB)
{
    Mat mat(path);
    return getImgDataF32C3(mat, size, needShuffleRGB);
}

std::vector<float> CVUtil::getImgDataF32C3(Mat &mat, const Vec2I32 &size, const bool &needShuffleRGB)
{
    if(mat.isEmpty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    MatOp::resize(mat, mat, size);

	if (mat.getChannel() == 1)
		MatOp::cvtColor(mat, mat, Msnhnet::CVT_GRAY2RGB);

    if(needShuffleRGB)
        MatOp::cvtColor(mat,mat,CVT_RGB2BGR);

    int width   = mat.getWidth();
    int height  = mat.getHeight();
    int channel = mat.getChannel();
    int step    = mat.getStep();

    std::vector<float> imgs(static_cast<size_t>(width*height*channel));

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int y = 0; y < height; ++y)
    {
        for (int k = 0; k < channel; ++k)
        {
            for (int x = 0; x < width; ++x)
            {
                imgs[static_cast<size_t>(k*width*height + y*width + x)] = mat.getData().u8[y*width*step + x*channel + k] / 255.0f;

            }
        }
    }

    mat.release();
    return imgs;
}

std::vector<float> CVUtil::getGoogLenetF32C3(const std::string &path, const Vec2I32 &size, const bool &needShuffleRGB)
{
    Mat mat(path);
    return getGoogLenetF32C3(mat, size, needShuffleRGB);
}

std::vector<float> CVUtil::getGoogLenetF32C3(Mat &mat, const Vec2I32 &size, const bool &needShuffleRGB)
{
    if(mat.isEmpty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    MatOp::resize(mat, mat, size);

    if (mat.getChannel() == 1)
        MatOp::cvtColor(mat, mat, Msnhnet::CVT_GRAY2RGB);

    if(needShuffleRGB)
        MatOp::cvtColor(mat,mat,CVT_RGB2BGR);

    int width   = mat.getWidth();
    int height  = mat.getHeight();
    int channel = mat.getChannel();
    int step    = mat.getStep();

    std::vector<float> imgs(static_cast<size_t>(width*height*channel));

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int y = 0; y < height; ++y)
    {
        for (int k = 0; k < channel; ++k)
        {
            for (int x = 0; x < width; ++x)
            {
                if(k == 0)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = mat.getData().u8[y*width*step + x*channel + k] / 255.0f * (0.229f / 0.5f) + (0.485f - 0.5f) / 0.5f;
                }
                else if(k == 1)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = mat.getData().u8[y*width*step + x*channel + k] / 255.0f * (0.224f / 0.5f) + (0.456f - 0.5f) / 0.5f;
                }
                else if(k == 2)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] =mat.getData().u8[y*width*step + x*channel + k] / 255.0f * (0.225f / 0.5f) + (0.406f - 0.5f) / 0.5f;
                }
            }
        }
    }

    mat.release();
    return imgs;
}

std::vector<float> CVUtil::getPaddingZeroF32C3(const std::string &path, const Vec2I32 &size, const bool &needShuffleRGB)
{
    Mat mat(path);
    return getPaddingZeroF32C3(mat, size, needShuffleRGB);
}

std::vector<float> CVUtil::getPaddingZeroF32C3(Mat &mat, const Vec2I32 &size, const bool &needShuffleRGB)
{
    if(mat.isEmpty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    int width   = mat.getWidth();
    int height  = mat.getHeight();
    int channel = mat.getChannel();

    std::vector<float> imgs(static_cast<size_t>(size.x1*size.x2*channel));

    int diff    =   abs(width - height);

    if(width > height)
    {
        MatOp::copyMakeBorder<Vec3U8>(mat, mat, diff/2, diff - diff/2, 0, 0, Vec3U8(127,127,127));
    }
    else if(width < height)
    {
        MatOp::copyMakeBorder<Vec3U8>(mat, mat,  0, 0, diff/2, diff - diff/2, Vec3U8(127,127,127));
    }

    MatOp::resize(mat, mat, size);

    if (mat.getChannel() == 1)
        MatOp::cvtColor(mat, mat, Msnhnet::CVT_GRAY2RGB);

    if(needShuffleRGB)
        MatOp::cvtColor(mat,mat,CVT_RGB2BGR);

    width   = mat.getWidth();
    height  = mat.getHeight();

    int step    = mat.getStep();

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int y = 0; y < height; ++y)
    {
        for (int k = 0; k < channel; ++k)
        {
            for (int x = 0; x < width; ++x)
            {
                imgs[static_cast<size_t>(k*width*height + y*width + x)] = mat.getData().u8[y*width*step + x*channel + k] / 255.0f;

            }
        }
    }

    mat.release();
    return imgs;
}

std::vector<float> CVUtil::getTransformedF32C3(const std::string &path, const Vec2I32 &size, const Vec3F32 &mean, const Vec3F32 &std, const bool &needShuffleRGB)
{
    Mat mat(path);
    return getTransformedF32C3(mat, size, mean, std, needShuffleRGB);
}

std::vector<float> CVUtil::getTransformedF32C3(Mat &mat, const Vec2I32 &size, const Vec3F32 &mean, const Vec3F32 &std, const bool &needShuffleRGB)
{
    if(mat.isEmpty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    MatOp::resize(mat, mat, size);

    if (mat.getChannel() == 1)
        MatOp::cvtColor(mat, mat, Msnhnet::CVT_GRAY2RGB);

    if(needShuffleRGB)
        MatOp::cvtColor(mat,mat,CVT_RGB2BGR);

    int width   = mat.getWidth();
    int height  = mat.getHeight();
    int channel = mat.getChannel();
    int step    = mat.getStep();

    std::vector<float> imgs(static_cast<size_t>(width*height*channel));

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int y = 0; y < height; ++y)
    {
        for (int k = 0; k < channel; ++k)
        {
            for (int x = 0; x < width; ++x)
            {
                if(k == 0)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = ((mat.getData().u8[y*width*step + x*channel + k] / 255.0f) - mean.x1)/std.x1;
                }
                else if(k == 1)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = ((mat.getData().u8[y*width*step + x*channel + k] / 255.0f) - mean.x2)/std.x2 ;
                }
                else if(k == 2)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = ((mat.getData().u8[y*width*step + x*channel + k] / 255.0f) - mean.x3)/std.x3 ;
                }
            }
        }
    }

    mat.release();
    return imgs;
}

std::vector<float> CVUtil::getCaffeModeF32C3(const std::string &path, const Vec2I32 &size, const bool &needShuffleRGB)
{
    Mat mat(path);
    return getCaffeModeF32C3(mat, size, needShuffleRGB);
}

std::vector<float> CVUtil::getCaffeModeF32C3(Mat &mat, const Vec2I32 &size, const bool &needShuffleRGB)
{
    if(mat.isEmpty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    MatOp::resize(mat, mat, size);

    if (mat.getChannel() == 1)
        MatOp::cvtColor(mat, mat, Msnhnet::CVT_GRAY2RGB);

    if(needShuffleRGB)
        MatOp::cvtColor(mat,mat,CVT_RGB2BGR);

    int width   = mat.getWidth();
    int height  = mat.getHeight();
    int channel = mat.getChannel();
    int step    = mat.getStep();

    std::vector<float> imgs(static_cast<size_t>(width*height*channel));

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int y = 0; y < height; ++y)
    {
        for (int k = 0; k < channel; ++k)
        {
            for (int x = 0; x < width; ++x)
            {
                if(k == 0)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = (mat.getData().u8[y*width*step + x*channel + k] ) - 123.68f  ;
                }
                else if(k == 1)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = (mat.getData().u8[y*width*step + x*channel + k]) - 116.779f ;
                }
                else if(k == 2)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = (mat.getData().u8[y*width*step + x*channel + k]) - 103.939f ;
                }
            }
        }
    }

    mat.release();
    return imgs;
}

void CVUtil::drawYoloBox(Mat &mat, std::vector<std::string> &labels, std::vector<std::vector<YoloBox> > &boxs, const Point2I &size, const bool &noPad)
{
    for (size_t i = 0; i < boxs[0].size(); ++i)
    {
        Msnhnet::YoloBox box;
        if(noPad)
            box= Msnhnet::YoloOutLayer::bboxResize2OrgNoPad(boxs[0][i],size,Msnhnet::Point2I(mat.getWidth(),mat.getHeight()));
        else
            box= Msnhnet::YoloOutLayer::bboxResize2Org(boxs[0][i],size,Msnhnet::Point2I(mat.getWidth(),mat.getHeight()));

        std::string label = std::to_string(static_cast<int>(box.conf*100)) + "% "+labels[static_cast<size_t>(box.bestClsIdx)];

        Draw::fillRect(mat,Vec2I32(static_cast<int>(box.xywhBox.x - box.xywhBox.w/2),static_cast<int>(box.xywhBox.y - box.xywhBox.h/2-32)),
                           Vec2I32(static_cast<int>(box.xywhBox.x - box.xywhBox.w/2 + label.length()*16),static_cast<int>(box.xywhBox.y - box.xywhBox.h/2))
                       ,CVUtil::colorTable[static_cast<size_t>(box.bestClsIdx)]);

        Draw::drawFont(mat,
                       label,
                       Vec2I32(static_cast<int>(box.xywhBox.x - box.xywhBox.w/2),
                               static_cast<int>(box.xywhBox.y - box.xywhBox.h/2-32)),
                       Vec3U8(255, 255, 255)
                       );

        float x = box.xywhBox.x;
        float y = box.xywhBox.y;
        float w = box.xywhBox.w;
        float h = box.xywhBox.h;

        float orgAngle = box.angle / 180.f*3.1415926f;
        float angle    = orgAngle;

        float v        = sqrtf(w*w + h*h)/2;
        float rad      = atan(h/w);

        float dx       = v*cosf(rad);
        float dy       = v*sinf(rad);

        float p1X = - dx;
        float p1Y = - dy;

        float p2X = + dx;
        float p2Y = - dy;

        float p3X = + dx;
        float p3Y = + dy;

        float p4X = - dx;
        float p4Y = + dy;

        float p1XF = p1X * cosf(angle)  + p1Y * sinf(angle);
        float p1YF = -p1X * sinf(angle) + p1Y * cosf(angle);

        float p2XF = p2X * cosf(angle)  + p2Y * sinf(angle);
        float p2YF = -p2X * sinf(angle) + p2Y * cosf(angle);

        float p3XF = p3X * cosf(angle)  + p3Y * sinf(angle);
        float p3YF = -p3X * sinf(angle) + p3Y * cosf(angle);

        float p4XF = p4X * cosf(angle)  + p4Y * sinf(angle);
        float p4YF = -p4X * sinf(angle) + p4Y * cosf(angle);

        Vec2I32 p1(static_cast<int>(p1XF+x),static_cast<int>(p1YF+y));
        Vec2I32 p2(static_cast<int>(p2XF+x),static_cast<int>(p2YF+y));
        Vec2I32 p3(static_cast<int>(p3XF+x),static_cast<int>(p3YF+y));
        Vec2I32 p4(static_cast<int>(p4XF+x),static_cast<int>(p4YF+y));

        Draw::drawRect(mat,p1,p2,p3,p4,CVUtil::colorTable[static_cast<size_t>(box.bestClsIdx)],3);
    }
}

void CVUtil::drawSegMask(const int &channel, const int &wxh, std::vector<float> &inVal, Mat &mask)
{
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < wxh; ++i)
    {
        float maxVal    = -FLT_MAX;
        int maxIndex  = 0;
        for (int j = 0; j < channel; ++j)
        {
            if(maxVal<(inVal[j*wxh + i]))
            {
                maxIndex = j;
                maxVal = inVal[j*wxh + i];
            }
        }

        Vec3U8 color = CVUtil::colorTable[maxIndex];

        mask.getData().u8[i*3+0] += static_cast<uint8_t>(color.x1/1.5F);
        mask.getData().u8[i*3+1] += static_cast<uint8_t>(color.x2/1.5F);
        mask.getData().u8[i*3+2] += static_cast<uint8_t>(color.x3/1.5F);
    }
}
}
