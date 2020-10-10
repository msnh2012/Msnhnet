#ifndef MSNHCVOP_H
#define MSNHCVOP_H

#include <Msnhnet/cv/MsnhCVMat.h>

namespace Msnhnet
{

class MatOp
{
public:
    static void getROI(Mat &src, Mat &dst, const Vec2I32 &p1, const Vec2I32 &p2);
    static void setROI(Mat &srcDst, Mat &roi, const Vec2I32 &pos);
    static void cvtColor(Mat &src, Mat &dst, const CvtColorType& cvtType);
    static void resize(Mat& src, Mat &dst, const Vec2I32 &outSize, const ResizeType& resizeType=RESIZE_BILINEAR);

    template<typename T>
    static void copyMakeBorder(Mat& src, Mat &dst, const int &top ,const int &down, const int &left, const int &right, const T & val)
    {
        int array   = DataType<T>::array;
        int fmt     = DataType<T>::fmt;
        src.checkPixelType(array, fmt);

        int srcWidth    = src.getWidth() ;
        int srcHeight   = src.getHeight();

        int finalWidth  = srcWidth + left + right;
        int finalHeight = srcHeight + top + down;

        Mat tmpMat(finalWidth, finalHeight, src.getMatType());
        tmpMat.fillPixel<T>(val);

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < srcHeight; ++i)
        {
            if(fmt == 'b')
            {
                memcpy(tmpMat.getData().u8+(finalWidth*(i+top)+left)*array, src.getData().u8+(srcWidth*i)*array, array*srcWidth);
            }
            else if(fmt == 'f')
            {
                memcpy(tmpMat.getData().f32+(finalWidth*(i+top)+left)*array, src.getData().f32+(srcWidth*i)*array, array*srcWidth*4);
            }
        }

        dst = tmpMat;
    }

private:
    static void RGB2BGR(const Mat &src, Mat &dst);
    static void RGB2GRAY(Mat &src, Mat &dst);
    static void RGBA2GRAY(Mat &src, Mat &dst);
    static void GRAY2RGB(Mat &src,Mat &dst);
    static void GRAY2RGBA(Mat &src, Mat &dst);
    static void RGB2RGBA(Mat &src, Mat &dst);
    static void RGBA2RGB(Mat &src, Mat &dst);
};

}

#endif 

