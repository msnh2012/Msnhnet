#include "Msnhnet/cv/MsnhCVMatOp.h"
namespace Msnhnet
{

void MatOp::roi(Mat &src, Mat &dst, const Vec2I32 &p1, const Vec2I32 &p2)
{

    if(src.isEmpty())
    {
        throw Exception(1,"[CV]: img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    int32_t width   = abs(p1.x1 - p2.x1);
    int32_t height  = abs(p1.x2 - p2.x2);
    int channel     = src.getChannel();
    MatType matType = src.getMatType();
    int step        = src.getStep();

    uint8_t* u8Ptr =  new uint8_t[dst.getWidth()*dst.getHeight()*dst.getStep()]();

    if(p1.x1 < 0 || p1.x2 < 0 || p1.x1 >= src.getWidth() || p1.x2>= src.getHeight() ||
            p2.x1 < 0 || p2.x2 < 0 || p2.x1 >= src.getWidth() || p2.x2>= src.getHeight()
            )
    {
        throw Exception(1,"[CV]: roi point pos out of memory", __FILE__, __LINE__, __FUNCTION__);
    }

    for (int i = 0; i < height; ++i)
    {
        memcpy(u8Ptr+i*width*dst.getStep(), src.getData().u8 + (p1.x2+i)*src.getWidth()*src.getStep() + p1.x1*src.getStep(),width*src.getStep());
    }

    dst.clearMat();
    dst.setChannel(channel);
    dst.setMatType(matType);
    dst.setStep(step);
    dst.setWidth(width);
    dst.setHeight(height);
    dst.setU8Ptr(u8Ptr);
}

void MatOp::cvtColor(Mat &src, Mat &dst, const CvtColorType &cvtType)
{
    if(src.isEmpty())
    {
        throw Exception(1,"[CV]: img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    switch (cvtType)
    {
    case CVT_RGB2GRAY:
        RGB2GRAY(src,dst);
        break;
    case CVT_RGBA2GRAY:
        RGB2GRAY(src,dst);
        break;
    case CVT_RGB2BGR:
        RGB2BGR(src,dst);
        break;
    default:
        break;
    }

}

void MatOp::RGB2BGR(const Mat &src, Mat &dst)
{

    if(src.getMatType() == MAT_GRAY_F32 || src.getMatType() == MAT_GRAY_U8)
    {
        throw Exception(1,"[CV]: RGB2BGR is not supported with single channel ", __FILE__, __LINE__, __FUNCTION__);
    }

    dst=src;

    int width   = dst.getWidth();
    int height  = dst.getHeight();

    switch (dst.getMatType())
    {
    case MAT_RGB_U8:
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                uint8_t tmp = dst.getData().u8[3*(i*width+j)+0];
                dst.getData().u8[3*(i*width+j)+0] = dst.getData().u8[3*(i*width+j)+2];
                dst.getData().u8[3*(i*width+j)+2] = tmp;
            }
        }
        break;
    case MAT_RGB_F32:
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                float tmp = dst.getData().f32[3*(i*width+j)+0];
                dst.getData().f32[3*(i*width+j)+0] = dst.getData().f32[3*(i*width+j)+2];
                dst.getData().f32[3*(i*width+j)+2] = tmp;
            }
        }
        break;
    case MAT_RGBA_U8:
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                uint8_t tmp = dst.getData().u8[4*(i*width+j)+0];
                dst.getData().u8[4*(i*width+j)+0] = dst.getData().u8[4*(i*width+j)+2];
                dst.getData().u8[4*(i*width+j)+2] = tmp;
            }
        }
        break;
    case MAT_RGBA_F32:
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                float tmp = dst.getData().f32[4*(i*width+j)+0];
                dst.getData().f32[4*(i*width+j)+0] = dst.getData().f32[4*(i*width+j)+2];
                dst.getData().f32[4*(i*width+j)+2] = tmp;
            }
        }
        break;
    }
}

void MatOp::RGB2GRAY(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_RGB_F32 && src.getMatType() != MAT_RGB_U8)
    {
        throw Exception(1,"[CV]: RGB2GRAY needs 3 channels ", __FILE__, __LINE__, __FUNCTION__);
    }

    int width   = src.getWidth();
    int height  = src.getHeight();

    MatData dstData;
    if(src.getMatType() == MAT_RGB_U8)
    {
        const int B = static_cast<int>(0.114 * 256 + 0.5);
        const int G = static_cast<int>(0.587 * 256 + 0.5);
        const int R = 256 - B - G;

        dstData.u8 = new uint8_t[width*height]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = 3*(i*width+j);
                dstData.u8[i*width+j] = (R*src.getData().u8[pos+0]+G*src.getData().u8[pos+1]+B*src.getData().u8[pos+2])>>8;
            }
        }

        dst.clearMat();
        dst.setChannel(1);
        dst.setMatType(MAT_GRAY_U8);
        dst.setStep(1);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
    else if(src.getMatType() == MAT_RGB_F32)
    {
        const float B = 0.114f;
        const float G = 0.587f;
        const float R = 1.f - B - G;

        dstData.u8 = new uint8_t[width*height*4]();

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = 3*(i*width+j);
                dstData.f32[i*width+j] = (R*src.getData().f32[pos+0]+G*src.getData().f32[pos+1]+B*src.getData().f32[pos+2]);
            }
        }
        dst.clearMat();
        dst.setChannel(1);
        dst.setMatType(MAT_GRAY_F32);
        dst.setStep(4);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }

}

void MatOp::RGBA2GRAY(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_RGBA_F32 && src.getMatType() != MAT_RGBA_U8)
    {
        throw Exception(1,"[CV]: RGBA2GRAY needs 4 channels ", __FILE__, __LINE__, __FUNCTION__);
    }

    int width   = src.getWidth();
    int height  = src.getHeight();

    MatData dstData;
    if(src.getMatType() == MAT_RGBA_U8)
    {
        const int B = static_cast<int>(0.114 * 256 + 0.5);
        const int G = static_cast<int>(0.587 * 256 + 0.5);
        const int R = 256 - B - G;

        dstData.u8 = new uint8_t[width*height]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = 4*(i*width+j);
                dstData.u8[i*width+j] = (R*src.getData().u8[pos+0]+G*src.getData().u8[pos+1]+B*src.getData().u8[pos+2])>>8;
            }
        }

        dst.clearMat();
        dst.setChannel(1);
        dst.setMatType(MAT_GRAY_U8);
        dst.setStep(1);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
    else if(src.getMatType() == MAT_RGBA_F32)
    {
        const float B = 0.114f;
        const float G = 0.587f;
        const float R = 1.f - B - G;

        dstData.u8 = new uint8_t[width*height*4]();

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = 4*(i*width+j);
                dstData.f32[i*width+j] = (R*src.getData().f32[pos+0]+G*src.getData().f32[pos+1]+B*src.getData().f32[pos+2]);
            }
        }
        dst.clearMat();
        dst.setChannel(1);
        dst.setMatType(MAT_GRAY_F32);
        dst.setStep(4);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

}
