#ifndef MSNHCVOP_H
#define MSNHCVOP_H

#include <Msnhnet/cv/MsnhCVMat.h>
#include <Msnhnet/utils/MsnhTimeUtil.h>

namespace Msnhnet
{

class MsnhNet_API MatOp
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
            else if(fmt == 'd')
            {
                memcpy(tmpMat.getData().f64+(finalWidth*(i+top)+left)*array, src.getData().f64+(srcWidth*i)*array, array*srcWidth*8);
            }
        }

        dst = tmpMat;
    }

    static void flip(Mat &mat, const FlipMode &flipMode=FLIP_V);

    static double norm(Mat &mat1, Mat &mat2, const NormType& normType = NORM_L2);
    static double norm(Mat &mat, const NormType& normType = NORM_L2);

    template<typename T>
    static void _split(Mat &src, std::vector<Mat> &dst)
    {
        dst.clear();

        if(src.getChannel()==1)
        {
            dst.push_back(src);
        }
        else if(src.getChannel()==3)
        {
            Mat R;
            Mat G;
            Mat B;

            Mat::createMat<T>(src.getWidth(),src.getHeight(),1,R);
            Mat::createMat<T>(src.getWidth(),src.getHeight(),1,G);
            Mat::createMat<T>(src.getWidth(),src.getHeight(),1,B);

#ifdef USE_OMP
            uint64_t dataLen   = src.getHeight()*src.getWidth();
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < src.getHeight(); ++i)
            {
                for (int j = 0; j < src.getWidth(); ++j)
                {
                    reinterpret_cast<T*>(R.getBytes())[i*src.getWidth() + j] = reinterpret_cast<T*>(src.getBytes())[(i*src.getWidth() + j)*3 + 0];
                    reinterpret_cast<T*>(G.getBytes())[i*src.getWidth() + j] = reinterpret_cast<T*>(src.getBytes())[(i*src.getWidth() + j)*3 + 1];
                    reinterpret_cast<T*>(B.getBytes())[i*src.getWidth() + j] = reinterpret_cast<T*>(src.getBytes())[(i*src.getWidth() + j)*3 + 2];
                }
            }
            dst.push_back(R);
            dst.push_back(G);
            dst.push_back(B);
        }
        else if(src.getChannel()==4)
        {
            Mat R;
            Mat G;
            Mat B;
            Mat A;

            Mat::createMat<T>(src.getWidth(),src.getHeight(),1,R);
            Mat::createMat<T>(src.getWidth(),src.getHeight(),1,G);
            Mat::createMat<T>(src.getWidth(),src.getHeight(),1,B);
            Mat::createMat<T>(src.getWidth(),src.getHeight(),1,A);

#ifdef USE_OMP
            uint64_t dataLen   = src.getHeight()*src.getWidth();
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < src.getHeight(); ++i)
            {
                for (int j = 0; j < src.getWidth(); ++j)
                {
                    reinterpret_cast<T*>(R.getBytes())[i*src.getWidth() + j] = reinterpret_cast<T*>(src.getBytes())[(i*src.getWidth() + j)*4 + 0];
                    reinterpret_cast<T*>(G.getBytes())[i*src.getWidth() + j] = reinterpret_cast<T*>(src.getBytes())[(i*src.getWidth() + j)*4 + 1];
                    reinterpret_cast<T*>(B.getBytes())[i*src.getWidth() + j] = reinterpret_cast<T*>(src.getBytes())[(i*src.getWidth() + j)*4 + 2];
                    reinterpret_cast<T*>(A.getBytes())[i*src.getWidth() + j] = reinterpret_cast<T*>(src.getBytes())[(i*src.getWidth() + j)*4 + 3];
                }
            }
            dst.push_back(R);
            dst.push_back(G);
            dst.push_back(B);
            dst.push_back(A);
        }
    }

    static void split(Mat &src, std::vector<Mat> &dst);

    template<typename T>
    static void _merge(std::vector<Mat> &src, Mat &dst)
    {

        int width  = src[0].getWidth();
        int height = src[1].getHeight();

        if(src.size()==1)
        {
            dst = src[0];
            return;
        }
        else if(src.size()==3)
        {
            Mat::createMat<T>(width,height,3,dst);

#ifdef USE_OMP
            uint64_t dataLen   = width*height;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    reinterpret_cast<T*>(dst.getBytes())[(i*width + j)*3 + 0] =  reinterpret_cast<T*>(src[0].getBytes())[i*width + j];
                    reinterpret_cast<T*>(dst.getBytes())[(i*width + j)*3 + 1] =  reinterpret_cast<T*>(src[1].getBytes())[i*width + j];
                    reinterpret_cast<T*>(dst.getBytes())[(i*width + j)*3 + 2] =  reinterpret_cast<T*>(src[2].getBytes())[i*width + j];
                }
            }
        }
        else if(src.size()==4)
        {
            Mat::createMat<T>(width,height,4,dst);

#ifdef USE_OMP
            uint64_t dataLen   = width*height;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    reinterpret_cast<T*>(dst.getBytes())[(i*width + j)*4 + 0] =  reinterpret_cast<T*>(src[0].getBytes())[i*width + j];
                    reinterpret_cast<T*>(dst.getBytes())[(i*width + j)*4 + 1] =  reinterpret_cast<T*>(src[1].getBytes())[i*width + j];
                    reinterpret_cast<T*>(dst.getBytes())[(i*width + j)*4 + 2] =  reinterpret_cast<T*>(src[2].getBytes())[i*width + j];
                    reinterpret_cast<T*>(dst.getBytes())[(i*width + j)*4 + 3] =  reinterpret_cast<T*>(src[3].getBytes())[i*width + j];
                }
            }
        }
    }

    static void merge(std::vector<Mat> &src, Mat &dst);

    static bool checkMatsProps(Mat &mat1, Mat &mat2);

    static void threshold(Mat &src, Mat &dst, const double& threshold, const double& maxVal, const int &thresholdType);

    static Mat hContact(const Mat &A, const Mat &B);

    static Mat vContact(const Mat &A, const Mat &B);

    static std::vector<int> histogram(Mat &src);

    static uint8_t getOtsu(Mat &src);

private:
    static void RGB2BGR(const Mat &src, Mat &dst);
    static void RGB2GRAY(Mat &src, Mat &dst);
    static void RGBA2GRAY(Mat &src, Mat &dst);
    static void GRAY2RGB(Mat &src,Mat &dst);
    static void GRAY2RGBA(Mat &src, Mat &dst);
    static void RGB2RGBA(Mat &src, Mat &dst);
    static void RGBA2RGB(Mat &src, Mat &dst);
    static void flipV(Mat &mat);
    static void flipH(Mat &mat);
};

}

#endif 

