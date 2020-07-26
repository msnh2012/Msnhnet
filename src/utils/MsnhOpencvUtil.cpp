#include "Msnhnet/utils/MsnhOpencvUtil.h"
namespace Msnhnet
{

std::vector<cv::Scalar> OpencvUtil::colorTable =  {
    cv::Scalar(0  , 0   ,200), cv::Scalar(0  , 200,   0), cv::Scalar(200 , 0 ,   0),
    cv::Scalar(0  , 255 ,200), cv::Scalar(255, 200,   0), cv::Scalar(200 , 0 , 255),
    cv::Scalar(50 , 0   ,200), cv::Scalar(50 , 200,   0), cv::Scalar(200 , 50,  50),
    cv::Scalar(50 , 255 ,200), cv::Scalar(255, 200,  50), cv::Scalar(200 , 50, 255),
    cv::Scalar(100, 0   ,200), cv::Scalar(100, 200,   0), cv::Scalar(200 ,100,  50),
    cv::Scalar(100, 255 ,200), cv::Scalar(255, 200, 100), cv::Scalar(200 ,100, 255),
    cv::Scalar(150, 0   ,200), cv::Scalar(150, 200,   0), cv::Scalar(200 ,150,  50),
    cv::Scalar(150, 255 ,200), cv::Scalar(255, 200, 150), cv::Scalar(200 ,150, 255),
    cv::Scalar(200, 0   ,200), cv::Scalar(200, 200,   0), cv::Scalar(200 ,200,  50),
    cv::Scalar(200, 255 ,200), cv::Scalar(255, 200, 200), cv::Scalar(200 ,200, 255),
    cv::Scalar(0  , 0   ,150), cv::Scalar(0  , 150,   0), cv::Scalar(150 , 0 ,   0),
    cv::Scalar(0  , 255 ,150), cv::Scalar(255, 150,   0), cv::Scalar(150 , 0 , 255),
    cv::Scalar(50 , 0   ,150), cv::Scalar(50 , 150,   0), cv::Scalar(150 , 50,  50),
    cv::Scalar(50 , 255 ,150), cv::Scalar(255, 150,  50), cv::Scalar(150 , 50, 255),
    cv::Scalar(100, 0   ,150), cv::Scalar(100, 150,   0), cv::Scalar(150 ,100,  50),
    cv::Scalar(100, 255 ,150), cv::Scalar(255, 150, 100), cv::Scalar(150 ,100, 255),
    cv::Scalar(150, 0   ,150), cv::Scalar(150, 150,   0), cv::Scalar(150 ,150,  50),
    cv::Scalar(150, 255 ,150), cv::Scalar(255, 150, 150), cv::Scalar(150 ,150, 255),
    cv::Scalar(200, 0   ,150), cv::Scalar(200, 150,   0), cv::Scalar(150 ,200,  50),
    cv::Scalar(200, 255 ,150), cv::Scalar(255, 150, 200), cv::Scalar(150 ,200, 255),
    cv::Scalar(0  , 0   ,255), cv::Scalar(0  , 255,   0), cv::Scalar(255 , 0 ,   0),
    cv::Scalar(0  , 255 ,255), cv::Scalar(255, 255,   0), cv::Scalar(255 , 0 , 255),
    cv::Scalar(50 , 0   ,255), cv::Scalar(50 , 255,   0), cv::Scalar(255 , 50,  50),
    cv::Scalar(50 , 255 ,255), cv::Scalar(255, 255,  50), cv::Scalar(255 , 50, 255),
    cv::Scalar(100, 0   ,255), cv::Scalar(100, 255,   0), cv::Scalar(255 ,100,  50),
    cv::Scalar(100, 255 ,255), cv::Scalar(255, 255, 100), cv::Scalar(255 ,100, 255),
    cv::Scalar(150, 0   ,255), cv::Scalar(150, 255,   0), cv::Scalar(255 ,150,  50),
    cv::Scalar(150, 255 ,255), cv::Scalar(255, 255, 150), cv::Scalar(255 ,150, 255),
    cv::Scalar(200, 0   ,255), cv::Scalar(200, 255,   0), cv::Scalar(255 ,200,  50),
    cv::Scalar(200, 255 ,255), cv::Scalar(255, 255, 200), cv::Scalar(255 ,200, 255),
};

OpencvUtil::OpencvUtil()
{
}

std::vector<float> OpencvUtil::getImgDataF32C1(const std::string &path, const cv::Size &size)
{

    cv::Mat mat = cv::imread(path.data());
    return getImgDataF32C1(mat, size);

}

std::vector<float> OpencvUtil::getImgDataF32C1(cv::Mat &mat, const cv::Size &size)
{
    if(mat.empty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__);
    }

    std::vector<float> imgs(static_cast<size_t>(mat.rows*mat.cols));

    cv::resize(mat, mat, size);

    if(mat.channels()==3)
    {
        cv::cvtColor(mat,mat,cv::COLOR_RGB2GRAY);
    }

    int width = mat.cols;
    int height = mat.rows;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            imgs[static_cast<size_t>(y*width + x)] = mat.data[y*width + x ] / 256.0f;
        }
    }
    mat.release();
    return imgs;
}

std::vector<float> OpencvUtil::getImgDataF32C3(const std::string &path, const cv::Size &size)
{
    cv::Mat mat = cv::imread(path.data());
    return getImgDataF32C3(mat, size);
}

std::vector<float> OpencvUtil::getImgDataF32C3(cv::Mat &mat, const cv::Size &size)
{
    if(mat.empty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__);
    }

    cv::resize(mat, mat, size);

    std::vector<float> imgs(static_cast<size_t>(mat.rows*mat.cols*3));

    int width   = mat.cols;
    int height  = mat.rows;
    int channel = mat.channels();

    int step    = static_cast<int>(mat.step);

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int y = 0; y < height; ++y)
    {
        for (int k = 0; k < channel; ++k)
        {
            for (int x = 0; x < width; ++x)
            {
                imgs[static_cast<size_t>(k*width*height + y*width + x)] = mat.data[y*step + x*channel + k] / 255.0f;

            }
        }
    }

    mat.release();
    return imgs;
}

std::vector<float> OpencvUtil::getGoogLenetF32C3(const std::string &path, const cv::Size &size)
{
    cv::Mat mat = cv::imread(path.data());
    return getGoogLenetF32C3(mat, size);
}

std::vector<float> OpencvUtil::getGoogLenetF32C3(cv::Mat &mat, const cv::Size &size)
{
    if(mat.empty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__);
    }

    cv::resize(mat, mat, size);

    std::vector<float> imgs(static_cast<size_t>(mat.rows*mat.cols*3));

    int width   = mat.cols;
    int height  = mat.rows;
    int channel = mat.channels();

    int step    = static_cast<int>(mat.step);

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
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = mat.data[y*step + x*channel + k] / 255.0f * (0.229f / 0.5f) + (0.485f - 0.5f) / 0.5f;
                }
                else if(k == 1)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = mat.data[y*step + x*channel + k] / 255.0f * (0.224f / 0.5f) + (0.456f - 0.5f) / 0.5f;
                }
                else if(k == 2)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = mat.data[y*step + x*channel + k] / 255.0f * (0.225f / 0.5f) + (0.406f - 0.5f) / 0.5f;
                }
            }
        }
    }

    mat.release();
    return imgs;
}

std::vector<float> OpencvUtil::getPaddingZeroF32C3(const std::string &path, const cv::Size &size)
{
    cv::Mat mat = cv::imread(path.data());
    return getPaddingZeroF32C3(mat, size);
}

std::vector<float> OpencvUtil::getPaddingZeroF32C3(cv::Mat &mat, const cv::Size &size)
{
    if(mat.empty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__);
    }

    int width   = mat.cols;
    int height  = mat.rows;
    int channel = mat.channels();

    std::vector<float> imgs(static_cast<size_t>(size.height*size.width*3));

    int diff    =   abs(width - height);

    if(width > height)
    {
        cv::copyMakeBorder(mat, mat, diff/2, diff - diff/2, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(127,127,127));
    }
    else if(width < height)
    {
        cv::copyMakeBorder(mat, mat, 0, 0, diff/2, diff - diff/2, cv::BORDER_CONSTANT, cv::Scalar(127,127,127));
    }

    cv::resize(mat, mat, size);

    cv::cvtColor(mat,mat,cv::COLOR_RGB2BGR);

    width   = mat.cols;
    height  = mat.rows;

    int step    = static_cast<int>(mat.step);

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int y = 0; y < height; ++y)
    {
        for (int k = 0; k < channel; ++k)
        {
            for (int x = 0; x < width; ++x)
            {
                imgs[static_cast<size_t>(k*width*height + y*width + x)] = mat.data[y*step + x*channel + k] / 255.0f;

            }
        }
    }

    mat.release();
    return imgs;

}

std::vector<float> OpencvUtil::getTransformedF32C3(const string &path, const cv::Size &size, const cv::Scalar &mean, const cv::Scalar &std)
{
    cv::Mat mat = cv::imread(path.data());
    return getTransformedF32C3(mat, size, mean, std);
}

std::vector<float> OpencvUtil::getTransformedF32C3(cv::Mat &mat, const cv::Size &size, const cv::Scalar &mean, const cv::Scalar &std)
{
    if(mat.empty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__);
    }

    cv::resize(mat, mat, size);

    std::vector<float> imgs(static_cast<size_t>(mat.rows*mat.cols*3));

    int width   = mat.cols;
    int height  = mat.rows;
    int channel = mat.channels();

    int step    = static_cast<int>(mat.step);

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
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = ((mat.data[y*step + x*channel + k] / 255.0f) - mean[0])/std[0];
                }
                else if(k == 1)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = ((mat.data[y*step + x*channel + k] / 255.0f) - mean[1])/std[1] ;
                }
                else if(k == 2)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = ((mat.data[y*step + x*channel + k] / 255.0f) - mean[2])/std[2] ;
                }
            }
        }
    }

    mat.release();
    return imgs;
}

std::vector<float> OpencvUtil::getCaffeModeF32C3(const string &path, const cv::Size &size)
{
    cv::Mat mat = cv::imread(path.data());
    return getCaffeModeF32C3(mat, size);
}

std::vector<float> OpencvUtil::getCaffeModeF32C3(cv::Mat &mat, const cv::Size &size)
{
    if(mat.empty())
    {
        throw Exception(1,"img empty", __FILE__, __LINE__);
    }

    cv::resize(mat, mat, size);

    std::vector<float> imgs(static_cast<size_t>(mat.rows*mat.cols*3));

    int width   = mat.cols;
    int height  = mat.rows;
    int channel = mat.channels();

    int step    = static_cast<int>(mat.step);

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
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = (mat.data[y*step + x*channel + k] ) - 123.68  ;
                }
                else if(k == 1)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = (mat.data[y*step + x*channel + k]) - 116.779 ;
                }
                else if(k == 2)
                {
                    imgs[static_cast<size_t>(k*width*height + y*width + x)] = (mat.data[y*step + x*channel + k]) - 103.939 ;
                }
            }
        }
    }

    mat.release();
    return imgs;
}

void OpencvUtil::drawYolov3Box(cv::Mat &mat, std::vector<std::string> &labels, std::vector<std::vector<Yolov3Box>> &boxs, const Point2I &size)
{
    for (size_t i = 0; i < boxs[0].size(); ++i)
    {
        Msnhnet::Yolov3Box box = Msnhnet::Yolov3OutLayer::bboxResize2org(boxs[0][i],size,Msnhnet::Point2I(mat.cols,mat.rows));

        std::string label = std::to_string(static_cast<int>(box.conf*100)) + "% "+labels[static_cast<size_t>(box.bestClsIdx)];

        cv::rectangle(mat,cv::Point(static_cast<int>(box.xywhBox.x - box.xywhBox.w/2),static_cast<int>(box.xywhBox.y - box.xywhBox.h/2-20)),
                      cv::Point(static_cast<int>(box.xywhBox.x - box.xywhBox.w/2 + label.length()*12),static_cast<int>(box.xywhBox.y - box.xywhBox.h/2)),
                      Msnhnet::OpencvUtil::colorTable[static_cast<size_t>(box.bestClsIdx)],-2,cv::LineTypes::LINE_AA);

        cv::RotatedRect rotRect(cv::Point2f(box.xywhBox.x,box.xywhBox.y),cv::Size2f(box.xywhBox.w,box.xywhBox.h),box.angle);
        cv::Point2f ver[4];
        rotRect.points(ver);
        for (int i = 0; i < 4; i++)
        {
            cv::line(mat, ver[i], ver[(i + 1) % 4], Msnhnet::OpencvUtil::colorTable[static_cast<size_t>(box.bestClsIdx)],2,cv::LineTypes::LINE_AA);
        }

        cv::putText(mat,
                    label,
                    cv::Point(static_cast<int>(box.xywhBox.x - box.xywhBox.w/2),
                              static_cast<int>(box.xywhBox.y - box.xywhBox.h/2 - 2)),
                    0, 0.6, cv::Scalar(255, 255, 255), 1,cv::LineTypes::LINE_AA);
    }
}
}
