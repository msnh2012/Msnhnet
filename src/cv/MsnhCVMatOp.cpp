#include "Msnhnet/cv/MsnhCVMatOp.h"
namespace Msnhnet
{

void MatOp::getROI(Mat &src, Mat &dst, const Vec2I32 &p1, const Vec2I32 &p2)
{
    if(src.isEmpty())
    {
        throw Exception(1,"[CV]: img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    int32_t width   = std::abs(p1.x1 - p2.x1);
    int32_t height  = std::abs(p1.x2 - p2.x2);
    int channel     = src.getChannel();
    MatType matType = src.getMatType();
    int step        = src.getStep();

    uint8_t* u8Ptr =  new uint8_t[width*height*step]();
    uint8_t* srcU8 =  src.getData().u8;

    if(p1.x1 < 0 || p1.x2 < 0 || p1.x1 >= src.getWidth() || p1.x2>= src.getHeight() ||
            p2.x1 < 0 || p2.x2 < 0 || p2.x1 >= src.getWidth() || p2.x2>= src.getHeight()
            )
    {
        throw Exception(1,"[CV]: roi point pos out of memory", __FILE__, __LINE__, __FUNCTION__);
    }

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < height; ++i)
    {
        memcpy(u8Ptr+i*width*step, srcU8 + (p1.x2+i)*src.getWidth()*step + p1.x1*step,width*step);
    }

    dst.release();
    dst.setChannel(channel);
    dst.setMatType(matType);
    dst.setStep(step);
    dst.setWidth(width);
    dst.setHeight(height);
    dst.setU8Ptr(u8Ptr);
}

void MatOp::setROI(Mat &srcDst, Mat &roi, const Vec2I32 &pos)
{
    if(srcDst.isEmpty())
    {
        throw Exception(1,"[CV]: img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    if(roi.isEmpty())
    {
        throw Exception(1,"[CV]: roi empty", __FILE__, __LINE__, __FUNCTION__);
    }

    if(srcDst.getMatType() != roi.getMatType())
    {
        throw Exception(1,"[CV]: roi mat type doesn't match src", __FILE__, __LINE__, __FUNCTION__);
    }
    int roiWidth    = roi.getWidth();
    int roiHeight   = roi.getHeight();

    int srcWidth    = srcDst.getWidth();
    int srcHeight   = srcDst.getHeight();

    int finalWidth  = roiWidth;
    int finalHeight = roiHeight;

    if((pos.x1 + roiWidth)>=srcWidth)
        finalWidth = srcWidth - pos.x1;

    if((pos.x2 + roiHeight)>=srcHeight)
        finalHeight = srcHeight - pos.x2;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < finalHeight; ++i)
    {
        memcpy(srcDst.getData().u8+((pos.x1+i)*srcWidth+pos.x2)*srcDst.getStep(), roi.getData().u8 + (i*roiWidth)*roi.getStep() ,finalWidth*roi.getStep());
    }

}

void MatOp::cvtColor(Mat &src, Mat &dst, const CvtColorType &cvtType)
{
    if(src.isEmpty())
    {
        throw Exception(1,"[CV]: img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    switch (cvtType)
    {
    case CVT_RGB2BGR:
        RGB2BGR(src,dst);
        break;
    case CVT_RGB2GRAY:
        RGB2GRAY(src,dst);
        break;
    case CVT_RGBA2GRAY:
        RGB2GRAY(src,dst);
        break;
    case CVT_GRAY2RGB:
        GRAY2RGB(src,dst);
        break;
    case CVT_GRAY2RGBA:
        GRAY2RGBA(src,dst);
        break;
    case CVT_RGB2RGBA:
        RGB2RGBA(src,dst);
        break;
    case CVT_RGBA2RGB:
        RGBA2RGB(src,dst);
        break;
    }

}

void MatOp::resize(Mat &src, Mat &dst, const Vec2I32 &outSize, const ResizeType &resizeType)
{
    if(src.isEmpty())
    {
        throw Exception(1,"[CV]: img empty", __FILE__, __LINE__, __FUNCTION__);
    }

    int srcWidth    = src.getWidth();
    int srcHeight   = src.getHeight();

    if(outSize.x1 == srcWidth && outSize.x2 == srcHeight)
    {
        dst = src;
        return;
    }

    const float fx = 1.f*srcWidth/outSize.x1;
    const float fy = 1.f*srcHeight/outSize.x2;
    const int srcStep        =  src.getStep();
    const MatType srcMatType =  src.getMatType();
    const int channel        =  src.getChannel();

    switch (resizeType)
    {
    case RESIZE_NEAREST:
    {

        uint8_t *u8Ptr      =  new uint8_t[outSize.x1*outSize.x2*srcStep]();
        uint8_t *srcU8      =  src.getData().u8;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < outSize.x2; ++i)
        {
            for (int j = 0; j < outSize.x1; ++j)
            {
                memcpy(u8Ptr+(i*outSize.x1+j)*srcStep, srcU8 + (static_cast<int>(i*fy)*srcWidth + static_cast<int>(j*fx))*srcStep, srcStep);
            }
        }

        dst.release();
        dst.setChannel(channel);
        dst.setWidth(outSize.x1);
        dst.setHeight(outSize.x2);
        dst.setMatType(srcMatType);
        dst.setStep(srcStep);
        dst.setU8Ptr(u8Ptr);

    }
        break;

    case RESIZE_BILINEAR:
    {

        MatData matData;
        matData.u8 = new uint8_t[outSize.x1*outSize.x2*srcStep]();

        switch (src.getMatType())
        {
        case MAT_GRAY_U8:
        {
            uint8_t* srcU8 = src.getData().u8;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < outSize.x2; ++i)
            {
                float srcIdxH = (i+0.5f)*fy-0.5f;
                srcIdxH = srcIdxH<0?0:srcIdxH;

                int srcIdxH0  = static_cast<int>(srcIdxH);
                int srcIdxH1  = std::min(srcIdxH0+1, srcHeight-1);

                for (int j = 0; j < outSize.x1; ++j)
                {
                    float srcIdxW = (j+0.5f)*fx-0.5f;
                    srcIdxW = srcIdxW<0?0:srcIdxW;

                    int srcIdxW0  = static_cast<int>(srcIdxW);
                    int srcIdxW1  = std::min(srcIdxW0+1, srcWidth-1);

                    uint16_t res  = static_cast<uint16_t>((srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW0)] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW1)])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW0)] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW1)]));

                    matData.u8[i*outSize.x1+j] = static_cast<uint8_t>(res>255?255:res);
                }
            }
        }
            break;
        case MAT_GRAY_F32:
        {
            float* srcF32 = src.getData().f32;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < outSize.x2; ++i)
            {
                float srcIdxH = (i+0.5f)*fy-0.5f;
                srcIdxH = srcIdxH<0?0:srcIdxH;

                int srcIdxH0  = static_cast<int>(srcIdxH);
                int srcIdxH1  = std::min(srcIdxH0+1, srcHeight-1);

                for (int j = 0; j < outSize.x1; ++j)
                {
                    float srcIdxW = (j+0.5f)*fx-0.5f;
                    srcIdxW = srcIdxW<0?0:srcIdxW;

                    int srcIdxW0  = static_cast<int>(srcIdxW);
                    int srcIdxW1  = std::min(srcIdxW0+1, srcWidth-1);

                    matData.f32[i*outSize.x1+j]  = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW0)] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW1)])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW0)] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW1)]);

                }
            }
        }
            break;
        case MAT_RGB_U8:
        {
            uint8_t* srcU8 = src.getData().u8;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < outSize.x2; ++i)
            {
                float srcIdxH = (i+0.5f)*fy-0.5f;
                srcIdxH = srcIdxH<0?0:srcIdxH;

                int srcIdxH0  = static_cast<int>(srcIdxH);
                int srcIdxH1  = std::min(srcIdxH0+1, srcHeight-1);

                for (int j = 0; j < outSize.x1; ++j)
                {
                    float srcIdxW = (j+0.5f)*fx-0.5f;
                    srcIdxW = srcIdxW<0?0:srcIdxW;

                    int srcIdxW0  = static_cast<int>(srcIdxW);
                    int srcIdxW1  = std::min(srcIdxW0+1, srcWidth-1);

                    uint16_t resR = static_cast<uint16_t>((srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+0])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+0]));

                    uint16_t resG = static_cast<uint16_t>((srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+1])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+1]));

                    uint16_t resB = static_cast<uint16_t>((srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+2])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+2]));

                    matData.u8[srcStep*(i*outSize.x1+j)+0] = static_cast<uint8_t>(resR>255?255:resR);
                    matData.u8[srcStep*(i*outSize.x1+j)+1] = static_cast<uint8_t>(resG>255?255:resG);
                    matData.u8[srcStep*(i*outSize.x1+j)+2] = static_cast<uint8_t>(resB>255?255:resB);
                }
            }
        }
            break;
        case MAT_RGB_F32:
        {
            float* srcF32 = src.getData().f32;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < outSize.x2; ++i)
            {
                float srcIdxH = (i+0.5f)*fy-0.5f;
                srcIdxH = srcIdxH<0?0:srcIdxH;

                int srcIdxH0  = static_cast<int>(srcIdxH);
                int srcIdxH1  = std::min(srcIdxH0+1, srcHeight-1);

                for (int j = 0; j < outSize.x1; ++j)
                {
                    float srcIdxW = (j+0.5f)*fx-0.5f;
                    srcIdxW = srcIdxW<0?0:srcIdxW;

                    int srcIdxW0  = static_cast<int>(srcIdxW);
                    int srcIdxW1  = std::min(srcIdxW0+1, srcWidth-1);

                    matData.f32[srcStep*(i*outSize.x1+j)+0] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+0])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+0]);

                    matData.f32[srcStep*(i*outSize.x1+j)+1] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+1])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+1]);

                    matData.f32[srcStep*(i*outSize.x1+j)+2] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+2])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+2]);

                }
            }
        }
            break;
        case MAT_RGBA_U8:
        {
            uint8_t* srcU8 = src.getData().u8;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < outSize.x2; ++i)
            {
                float srcIdxH = (i+0.5f)*fy-0.5f;
                srcIdxH = srcIdxH<0?0:srcIdxH;

                int srcIdxH0  = static_cast<int>(srcIdxH);
                int srcIdxH1  = std::min(srcIdxH0+1, srcHeight-1);

                for (int j = 0; j < outSize.x1; ++j)
                {
                    float srcIdxW = (j+0.5f)*fx-0.5f;
                    srcIdxW = srcIdxW<0?0:srcIdxW;

                    int srcIdxW0  = static_cast<int>(srcIdxW);
                    int srcIdxW1  = std::min(srcIdxW0+1, srcWidth-1);

                    uint16_t resR = static_cast<uint16_t>((srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+0])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+0]));

                    uint16_t resG = static_cast<uint16_t>((srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+1])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+1]));

                    uint16_t resB = static_cast<uint16_t>((srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+2])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+2]));

                    uint16_t resA = static_cast<uint16_t>((srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+3] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+3])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+3] + (srcIdxW-srcIdxW0)*srcU8[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+3]));

                    matData.u8[srcStep*(i*outSize.x1+j)+0] = static_cast<uint8_t>(resR>255?255:resR);
                    matData.u8[srcStep*(i*outSize.x1+j)+1] = static_cast<uint8_t>(resG>255?255:resG);
                    matData.u8[srcStep*(i*outSize.x1+j)+2] = static_cast<uint8_t>(resB>255?255:resB);
                    matData.u8[srcStep*(i*outSize.x1+j)+3] = static_cast<uint8_t>(resA>255?255:resA);
                }
            }
        }
            break;
        case MAT_RGBA_F32:
        {
            float* srcF32 = src.getData().f32;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < outSize.x2; ++i)
            {
                float srcIdxH = (i+0.5f)*fy-0.5f;
                srcIdxH = srcIdxH<0?0:srcIdxH;

                int srcIdxH0  = static_cast<int>(srcIdxH);
                int srcIdxH1  = std::min(srcIdxH0+1, srcHeight-1);

                for (int j = 0; j < outSize.x1; ++j)
                {
                    float srcIdxW = (j+0.5f)*fx-0.5f;
                    srcIdxW = srcIdxW<0?0:srcIdxW;

                    int srcIdxW0  = static_cast<int>(srcIdxW);
                    int srcIdxW1  = std::min(srcIdxW0+1, srcWidth-1);

                    matData.f32[srcStep*(i*outSize.x1+j)+0] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+0])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+0]);

                    matData.f32[srcStep*(i*outSize.x1+j)+1] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+1])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+1]);

                    matData.f32[srcStep*(i*outSize.x1+j)+2] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+2])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+2]);

                    matData.f32[srcStep*(i*outSize.x1+j)+3] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+3] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+3])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+3] + (srcIdxW-srcIdxW0)*srcF32[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+3]);

                }
            }
        }
        break;
        }
        dst.release();
        dst.setChannel(channel);
        dst.setWidth(outSize.x1);
        dst.setHeight(outSize.x2);
        dst.setMatType(srcMatType);
        dst.setStep(srcStep);
        dst.setU8Ptr(matData.u8);
    }
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
                int pos = 3*(i*width+j);
                uint8_t tmp = dst.getData().u8[pos+0];
                dst.getData().u8[pos+0] = dst.getData().u8[pos+2];
                dst.getData().u8[pos+2] = tmp;
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
                int pos = 3*(i*width+j);
                float tmp = dst.getData().f32[pos+0];
                dst.getData().f32[pos+0] = dst.getData().f32[pos+2];
                dst.getData().f32[pos+2] = tmp;
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
                int pos = 4*(i*width+j);
                uint8_t tmp = dst.getData().u8[pos+0];
                dst.getData().u8[pos+0] = dst.getData().u8[pos+2];
                dst.getData().u8[pos+2] = tmp;
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
                int pos = 4*(i*width+j);
                float tmp = dst.getData().f32[pos+0];
                dst.getData().f32[pos+0] = dst.getData().f32[pos+2];
                dst.getData().f32[pos+2] = tmp;
            }
        }
        break;
    }
}

void MatOp::RGB2GRAY(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_RGB_F32 && src.getMatType() != MAT_RGB_U8)
    {
        throw Exception(1,"[CV]: RGB2GRAY src needs 3 channels ", __FILE__, __LINE__, __FUNCTION__);
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
        uint8_t* srcU8 = src.getData().u8;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = i*width+j;
                dstData.u8[pos] = (R*srcU8[pos*3+0]+G*srcU8[pos*3+1]+B*srcU8[pos*3+2])>>8;
            }
        }

        dst.release();
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

        float* srcF32 = src.getData().f32;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = i*width+j;
                dstData.f32[pos] = (R*srcF32[3*pos+0]+G*srcF32[3*pos+1]+B*srcF32[3*pos+2]);
            }
        }
        dst.release();
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
        throw Exception(1,"[CV]: RGBA2GRAY src needs 4 channels ", __FILE__, __LINE__, __FUNCTION__);
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
        uint8_t* srcU8 = src.getData().u8;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = i*width+j;
                dstData.u8[pos] = (R*srcU8[pos*4+0]+G*srcU8[pos*4+1]+B*srcU8[pos*4+2])>>8;
            }
        }

        dst.release();
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
        float* srcF32 = src.getData().f32;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = i*width+j;
                dstData.f32[pos] = (R*srcF32[4*pos+0]+G*srcF32[4*pos+1]+B*srcF32[4*pos+2]);
            }
        }
        dst.release();
        dst.setChannel(1);
        dst.setMatType(MAT_GRAY_F32);
        dst.setStep(4);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

void MatOp::GRAY2RGB(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_GRAY_F32 && src.getMatType() != MAT_GRAY_U8)
    {
        throw Exception(1,"[CV]: GRAY2RGB src needs 1 channel ", __FILE__, __LINE__, __FUNCTION__);
    }

    int width   = src.getWidth();
    int height  = src.getHeight();

    MatData dstData;
    if(src.getMatType() == MAT_GRAY_U8)
    {
        dstData.u8 = new uint8_t[width*height*3]();
        uint8_t* srcU8 = src.getData().u8;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.u8[pos*3+0] = srcU8[pos];
                dstData.u8[pos*3+1] = srcU8[pos];
                dstData.u8[pos*3+2] = srcU8[pos];
            }
        }
        dst.release();
        dst.setChannel(3);
        dst.setMatType(MAT_RGB_U8);
        dst.setStep(3);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
    else if(src.getMatType() == MAT_GRAY_F32)
    {
        dstData.u8 = new uint8_t[width*height*12]();
        float* srcF32 = src.getData().f32;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.f32[pos*3+0] = srcF32[pos];
                dstData.f32[pos*3+1] = srcF32[pos];
                dstData.f32[pos*3+2] = srcF32[pos];
            }
        }
        dst.release();
        dst.setChannel(3);
        dst.setMatType(MAT_RGB_F32);
        dst.setStep(12);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

void MatOp::GRAY2RGBA(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_GRAY_F32 && src.getMatType() != MAT_GRAY_U8)
    {
        throw Exception(1,"[CV]: GRAY2RGBA src needs 1 channel ", __FILE__, __LINE__, __FUNCTION__);
    }

    int width   = src.getWidth();
    int height  = src.getHeight();

    MatData dstData;
    if(src.getMatType() == MAT_GRAY_U8)
    {
        dstData.u8 = new uint8_t[width*height*4]();
        uint8_t* srcU8 = src.getData().u8;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.u8[pos*4+0] = srcU8[pos];
                dstData.u8[pos*4+1] = srcU8[pos];
                dstData.u8[pos*4+2] = srcU8[pos];
                dstData.u8[pos*4+3] = 255;
            }
        }
        dst.release();
        dst.setChannel(4);
        dst.setMatType(MAT_RGBA_U8);
        dst.setStep(4);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
    else if(src.getMatType() == MAT_GRAY_F32)
    {
        dstData.u8 = new uint8_t[width*height*16]();
        float* srcF32 = src.getData().f32;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.f32[pos*4+0] = srcF32[pos];
                dstData.f32[pos*4+1] = srcF32[pos];
                dstData.f32[pos*4+2] = srcF32[pos];
                dstData.f32[pos*4+3] = 1.f;
            }
        }
        dst.release();
        dst.setChannel(4);
        dst.setMatType(MAT_RGBA_F32);
        dst.setStep(16);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

void MatOp::RGB2RGBA(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_RGB_F32 && src.getMatType() != MAT_RGB_U8)
    {
        throw Exception(1,"[CV]: RGB2RGBA src needs 3 channels ", __FILE__, __LINE__, __FUNCTION__);
    }

    int width   = src.getWidth();
    int height  = src.getHeight();

    MatData dstData;
    if(src.getMatType() == MAT_RGB_U8)
    {
        dstData.u8 = new uint8_t[width*height*4]();
        uint8_t* srcU8 = src.getData().u8;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.u8[pos*4+0] = srcU8[3*pos+0];
                dstData.u8[pos*4+1] = srcU8[3*pos+1];
                dstData.u8[pos*4+2] = srcU8[3*pos+2];
                dstData.u8[pos*4+3] = 255;
            }
        }
        dst.release();
        dst.setChannel(4);
        dst.setMatType(MAT_RGBA_U8);
        dst.setStep(4);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
    else if(src.getMatType() == MAT_RGB_F32)
    {
        dstData.u8 = new uint8_t[width*height*16]();
        float* srcF32 = src.getData().f32;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.f32[pos*4+0] = srcF32[3*pos+0];
                dstData.f32[pos*4+1] = srcF32[3*pos+1];
                dstData.f32[pos*4+2] = srcF32[3*pos+2];
                dstData.f32[pos*4+3] = 1.f;
            }
        }
        dst.release();
        dst.setChannel(4);
        dst.setMatType(MAT_RGBA_F32);
        dst.setStep(16);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

void MatOp::RGBA2RGB(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_RGBA_F32 && src.getMatType() != MAT_RGBA_U8)
    {
        throw Exception(1,"[CV]: RGBA2RGB src needs 4 channels ", __FILE__, __LINE__, __FUNCTION__);
    }

    int width   = src.getWidth();
    int height  = src.getHeight();

    MatData dstData;
    if(src.getMatType() == MAT_RGBA_U8)
    {
        dstData.u8 = new uint8_t[width*height*3]();
        uint8_t* srcU8 = src.getData().u8;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.u8[pos*3+0] = srcU8[pos*4+0];
                dstData.u8[pos*3+1] = srcU8[pos*4+1];
                dstData.u8[pos*3+2] = srcU8[pos*4+2];
            }
        }
        dst.release();
        dst.setChannel(3);
        dst.setMatType(MAT_RGB_U8);
        dst.setStep(3);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
    else if(src.getMatType() == MAT_RGBA_F32)
    {
        dstData.u8 = new uint8_t[width*height*12]();
        float* srcF32 = src.getData().f32;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.f32[pos*3+0] = srcF32[pos*4+0];
                dstData.f32[pos*3+1] = srcF32[pos*4+1];
                dstData.f32[pos*3+2] = srcF32[pos*4+2];
            }
        }
        dst.release();
        dst.setChannel(3);
        dst.setMatType(MAT_RGB_F32);
        dst.setStep(12);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

}
