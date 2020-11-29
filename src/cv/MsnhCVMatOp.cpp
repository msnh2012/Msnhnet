#include "Msnhnet/cv/MsnhCVMatOp.h"
namespace Msnhnet
{

void MatOp::getROI(Mat &src, Mat &dst, const Vec2I32 &p1, const Vec2I32 &p2)
{
    if(src.isEmpty())
    {
        throw Exception(1,"[MatOp]: img empty! \n", __FILE__, __LINE__, __FUNCTION__);
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
        throw Exception(1,"[MatOp]: roi point pos out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
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
        throw Exception(1,"[MatOp]: img empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(roi.isEmpty())
    {
        throw Exception(1,"[MatOp]: roi empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(srcDst.getMatType() != roi.getMatType())
    {
        throw Exception(1,"[MatOp]: roi mat type doesn't match src! \n", __FILE__, __LINE__, __FUNCTION__);
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
        throw Exception(1,"[MatOp]: img empty! \n", __FILE__, __LINE__, __FUNCTION__);
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
        throw Exception(1,"[MatOp]: img empty! \n", __FILE__, __LINE__, __FUNCTION__);
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
        case MAT_GRAY_F64:
        {
            double* srcF64 = src.getData().f64;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < outSize.x2; ++i)
            {
                double srcIdxH = (i+0.5)*fy-0.5;
                srcIdxH = srcIdxH<0?0:srcIdxH;

                int srcIdxH0  = static_cast<int>(srcIdxH);
                int srcIdxH1  = std::min(srcIdxH0+1, srcHeight-1);

                for (int j = 0; j < outSize.x1; ++j)
                {
                    double srcIdxW = (j+0.5)*fx-0.5;
                    srcIdxW = srcIdxW<0?0:srcIdxW;

                    int srcIdxW0  = static_cast<int>(srcIdxW);
                    int srcIdxW1  = std::min(srcIdxW0+1, srcWidth-1);

                    matData.f64[i*outSize.x1+j]  = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW0)] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW1)])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW0)] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW1)]);

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
        case MAT_RGB_F64:
        {
            double* srcF64 = src.getData().f64;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < outSize.x2; ++i)
            {
                double srcIdxH = (i+0.5)*fy-0.5;
                srcIdxH = srcIdxH<0?0:srcIdxH;

                int srcIdxH0  = static_cast<int>(srcIdxH);
                int srcIdxH1  = std::min(srcIdxH0+1, srcHeight-1);

                for (int j = 0; j < outSize.x1; ++j)
                {
                    double srcIdxW = (j+0.5)*fx-0.5;
                    srcIdxW = srcIdxW<0?0:srcIdxW;

                    int srcIdxW0  = static_cast<int>(srcIdxW);
                    int srcIdxW1  = std::min(srcIdxW0+1, srcWidth-1);

                    matData.f64[srcStep*(i*outSize.x1+j)+0] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+0])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+0]);

                    matData.f64[srcStep*(i*outSize.x1+j)+1] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+1])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+1]);

                    matData.f64[srcStep*(i*outSize.x1+j)+2] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+2])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+2]);

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
        case MAT_RGBA_F64:
        {
            double* srcF64 = src.getData().f64;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < outSize.x2; ++i)
            {
                double srcIdxH = (i+0.5)*fy-0.5;
                srcIdxH = srcIdxH<0?0:srcIdxH;

                int srcIdxH0  = static_cast<int>(srcIdxH);
                int srcIdxH1  = std::min(srcIdxH0+1, srcHeight-1);

                for (int j = 0; j < outSize.x1; ++j)
                {
                    double srcIdxW = (j+0.5)*fx-0.5;
                    srcIdxW = srcIdxW<0?0:srcIdxW;

                    int srcIdxW0  = static_cast<int>(srcIdxW);
                    int srcIdxW1  = std::min(srcIdxW0+1, srcWidth-1);

                    matData.f64[srcStep*(i*outSize.x1+j)+0] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+0])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+0] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+0]);

                    matData.f64[srcStep*(i*outSize.x1+j)+1] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+1])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+1] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+1]);

                    matData.f64[srcStep*(i*outSize.x1+j)+2] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+2])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+2] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+2]);

                    matData.f64[srcStep*(i*outSize.x1+j)+3] = (srcIdxH1-srcIdxH)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW0)+3] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH0*srcWidth+srcIdxW1)+3])+

                            (srcIdxH -srcIdxH0)*((srcIdxW1 - srcIdxW)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW0)+3] + (srcIdxW-srcIdxW0)*srcF64[srcStep*(srcIdxH1*srcWidth+srcIdxW1)+3]);

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

void MatOp::flip(Mat &mat, const FlipMode &flipMode)
{
    if(flipMode == FLIP_H)
    {
        flipH(mat);
    }
    else if(flipMode == FLIP_V)
    {
        flipV(mat);
    }
}

double MatOp::norm(Mat &mat1, Mat &mat2, const NormType &normType)
{
    if(!checkMatsProps(mat1,mat2))
    {
        throw Exception(1,"[MatOp]: mat1 props and mat2 props must be same ! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(mat1.isU8Mat())
    {
        mat1.convertTo(mat1, CVT_DATA_TO_F32_DIRECTLY);
        mat2.convertTo(mat2, CVT_DATA_TO_F32_DIRECTLY);
    }

    mat1 = mat1-mat2;

    return norm(mat1,normType);
}

double MatOp::norm(Mat &mat, const NormType &normType)
{
    if(mat.isU8Mat())
    {
        mat.convertTo(mat, CVT_DATA_TO_F32_DIRECTLY);
    }

    double final = 0;

    if(normType==NORM_L1)
    {
        if(mat.isF32Mat())
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:final)
#endif
            for (int i = 0; i < mat.getDataNum(); ++i)
            {
                final += std::abs(mat.getFloat32()[i]);
            }
        }
        else if(mat.isF64Mat())
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:final)
#endif
            for (int i = 0; i < mat.getDataNum(); ++i)
            {
                final += std::abs(mat.getFloat64()[i]);
            }
        }
        return final;
    }
    else if(normType==NORM_L2 || normType==NORM_L2_SQR)
    {
        if(mat.isF32Mat())
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:final)
#endif
            for (int i = 0; i < mat.getDataNum(); ++i)
            {
                float v = mat.getFloat32()[i];
                final += v*v;
            }
        }
        else if(mat.isF64Mat())
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:final)
#endif
            for (int i = 0; i < mat.getDataNum(); ++i)
            {
                double v = mat.getFloat64()[i];
                final += v*v;
            }
        }

        if(normType==NORM_L2_SQR)
        {
            return final;
        }
        else if(normType==NORM_L2)
        {
            return  std::sqrt(final);
        }
    }
    else if(normType==NORM_INF)
    {
#ifdef USE_OMP
        std::vector<double> tmpMax(OMP_THREAD,0);
#endif
        if(mat.isF32Mat())
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < mat.getDataNum(); ++i)
            {
                double v = std::abs(mat.getFloat32()[i]);
#ifdef USE_OMP
                if(v>tmpMax[omp_get_thread_num()])
                {
                    tmpMax[omp_get_thread_num()] = v;
                }
#else
                if(v>tmpMax[omp_get_thread_num()])
                {
                    final = v;
                }
#endif
            }
        }
        else if(mat.isF64Mat())
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < mat.getDataNum(); ++i)
            {
                double v = std::abs(mat.getFloat64()[i]);
#ifdef USE_OMP
                if(v>tmpMax[omp_get_thread_num()])
                {
                    tmpMax[omp_get_thread_num()] = v;
                }
#else
                if(v>tmpMax[omp_get_thread_num()])
                {
                    final = v;
                }
#endif
            }
        }
#ifdef USE_OMP
        return  *std::max_element(tmpMax.begin(), tmpMax.end());
#else
        return final;
#endif
    }
}

void MatOp::split(Mat &src, std::vector<Mat> &dst)
{

    if(src.isU8Mat())
    {
        _split<uint8_t>(src,dst);
    }
    else if(src.isF32Mat())
    {
        _split<float>(src,dst);
    }
    else if(src.isF64Mat())
    {
        _split<double>(src,dst);
    }

}

void MatOp::merge(std::vector<Mat> &src, Mat &dst)
{
    if(src.empty())
    {
        throw Exception(1,"[MatOp]: Merge src mats empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    for (int i = 0; i < src.size()-1; ++i)
    {
        if(!checkMatsProps(src[i],src[i+1]))
        {
            throw Exception(1,"[MatOp]: Merge mats props must be same! \n", __FILE__, __LINE__, __FUNCTION__);
        }
    }

    if(src[0].getChannel()!=1)
    {
        throw Exception(1,"[MatOp]: Merge mats must be only 1 channel! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(src.size()!=1 && src.size()!=3 && src.size()!=4)
    {
        throw Exception(1,"[MatOp]: Merge src mats size must be 1/3/4! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(src[0].isU8Mat())
    {
        _merge<uint8_t>(src,dst);
    }
    else if(src[0].isF32Mat())
    {
        _merge<float>(src,dst);
    }
    else if(src[0].isF64Mat())
    {
        _merge<double>(src,dst);
    }
}

bool MatOp::checkMatsProps(Mat &mat1, Mat &mat2)
{
    if(mat1.getMatType()!=mat2.getMatType() || mat1.getWidth()!=mat2.getWidth() || mat1.getHeight()!=mat2.getHeight())
    {
        return false;
    }

    return true;
}

void MatOp::threshold(Mat &src, Mat &dst, const double &threshold, const double &maxVal, const int &thresholdType)
{

    if(threshold > maxVal)
    {
        throw Exception(1,"[MatOp]: threshold should < maxVal ! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    int thType = thresholdType;

    if(src.isU8Mat())
    {
        if(threshold>255 || maxVal>255)
        {
            throw Exception(1,"[MatOp]: threshold and maxVal should < 256 ! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        uint8_t thU8  = static_cast<uint8_t>(threshold);
        uint8_t maxU8 = static_cast<uint8_t>(maxVal);

        if((thType>>3)==1)
        {
            if(src.getChannel()!=1)
                throw Exception(1,"[MatOp]: Otus channel should = 1 ! \n", __FILE__, __LINE__, __FUNCTION__);

            thU8 = getOtsu(src);

            std::cout<<(int)thU8<<"----------------------"<<std::endl;

            thType = thType - 8;
        }

        dst = src;

        if(thType == THRESH_BINARY)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getBytes()[i]>thU8)
                {
                    dst.getBytes()[i] = maxU8;
                }
                else
                {
                    dst.getBytes()[i] = 0;
                }
            }
        }
        else if(thType == THRESH_BINARY_INV)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getBytes()[i]>thU8)
                {
                    dst.getBytes()[i] = 0;
                }
                else
                {
                    dst.getBytes()[i] = maxU8;
                }
            }
        }
        else if(thType == THRESH_TOZERO)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getBytes()[i]<thU8)
                {
                    dst.getBytes()[i] = 0;
                }
            }
        }
        else if(thType == THRESH_TOZERO_INV)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getBytes()[i]>thU8)
                {
                    dst.getBytes()[i] = 0;
                }
            }
        }
    }
    else if(src.isF32Mat())
    {

        float thF32  = static_cast<float>(threshold);
        float maxF32 = static_cast<float>(maxVal);

        dst = src;

        if(thType == THRESH_BINARY)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getFloat32()[i]>thF32)
                {
                    dst.getFloat32()[i] = maxF32;
                }
                else
                {
                    dst.getFloat32()[i] = 0;
                }
            }
        }
        else if(thType == THRESH_BINARY_INV)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getFloat32()[i]>thF32)
                {
                    dst.getFloat32()[i] = 0;
                }
                else
                {
                    dst.getFloat32()[i] = maxF32;
                }
            }
        }
        else if(thType == THRESH_TOZERO)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getFloat32()[i]<thF32)
                {
                    dst.getFloat32()[i] = 0;
                }
            }
        }
        else if(thType == THRESH_TOZERO_INV)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getFloat32()[i]>thF32)
                {
                    dst.getFloat32()[i] = 0;
                }
            }
        }
    }
    else if(src.isF64Mat())
    {

        double thF64  = threshold;
        double maxF64 = maxVal;

        dst = src;

        if(thType == THRESH_BINARY)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getFloat64()[i]>thF64)
                {
                    dst.getFloat64()[i] = maxF64;
                }
                else
                {
                    dst.getFloat64()[i] = 0;
                }
            }
        }
        else if(thType == THRESH_BINARY_INV)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getFloat64()[i]>thF64)
                {
                    dst.getFloat64()[i] = 0;
                }
                else
                {
                    dst.getFloat64()[i] = maxF64;
                }
            }
        }
        else if(thType == THRESH_TOZERO)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getFloat64()[i]<thF64)
                {
                    dst.getFloat64()[i] = 0;
                }
            }
        }
        else if(thType == THRESH_TOZERO_INV)
        {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < src.getDataNum(); ++i)
            {
                if(src.getFloat64()[i]>thF64)
                {
                    dst.getFloat64()[i] = 0;
                }
            }
        }
    }
}

std::vector<int> MatOp::histogram(Mat &src)
{
    std::vector<int> hist(256,0);

    if(src.getMatType()!=MAT_GRAY_U8)
    {
        throw Exception(1,"[MatOp]: histogram mat type must be GRAY U8! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    for (int i = 0; i < src.getDataNum(); ++i)
    {
        uint8_t val = src.getBytes()[i];
        hist[val] += 1;
    }

    return hist;
}

uint8_t MatOp::getOtsu(Mat &src)
{
    if(src.getMatType()!=MAT_GRAY_U8)
    {
        throw Exception(1,"[MatOp]: histogram mat type must be GRAY U8! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    int threshold;

    int height = src.getHeight();
    int width  = src.getWidth();

    std::vector<int> hist = histogram(src);

    std::vector<float> histF(256,0);

    int size = height*width;
    for (int i = 0; i < 256; i++)
    {
        histF[i] = 1.0f*hist[i] / size;
    }

    float avgValue = 0;
    for (int i = 0; i < 256; i++)
    {
        avgValue += i*histF[i];
    }

    float maxVariance = 0;
    float w = 0;
    float u = 0;
    for (int i = 0; i < 256; i++)
    {
        w += histF[i];
        u += i*histF[i];

        float t = avgValue*w - u;
        float variance = t*t / (w*(1 - w));

        if (variance > maxVariance)
        {
            maxVariance = variance;
            threshold = i;
        }
    }

    return static_cast<uint8_t>(threshold);

}

void MatOp::RGB2BGR(const Mat &src, Mat &dst)
{

    if(src.getMatType() == MAT_GRAY_F32 || src.getMatType() == MAT_GRAY_U8 || src.getMatType() == MAT_GRAY_F64)
    {
        throw Exception(1,"[MatOp]: RGB2BGR is not supported with single channel! \n", __FILE__, __LINE__, __FUNCTION__);
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
    case MAT_RGB_F64:
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = 3*(i*width+j);
                double tmp = dst.getData().f64[pos+0];
                dst.getData().f64[pos+0] = dst.getData().f64[pos+2];
                dst.getData().f64[pos+2] = tmp;
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
    case MAT_RGBA_F64:
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = 4*(i*width+j);
                double tmp = dst.getData().f64[pos+0];
                dst.getData().f64[pos+0] = dst.getData().f64[pos+2];
                dst.getData().f64[pos+2] = tmp;
            }
        }
        break;
    }
}

void MatOp::RGB2GRAY(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_RGB_F32 && src.getMatType() != MAT_RGB_U8 && src.getMatType() != MAT_RGB_F64)
    {
        throw Exception(1,"[MatOp]: RGB2GRAY src needs 3 channels! \n", __FILE__, __LINE__, __FUNCTION__);
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
    else if(src.getMatType() == MAT_RGB_F64)
    {
        const double B = 0.114;
        const double G = 0.587;
        const double R = 1 - B - G;

        dstData.u8 = new uint8_t[width*height*8]();

        double* srcF64 = src.getData().f64;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = i*width+j;
                dstData.f64[pos] = (R*srcF64[3*pos+0]+G*srcF64[3*pos+1]+B*srcF64[3*pos+2]);
            }
        }
        dst.release();
        dst.setChannel(1);
        dst.setMatType(MAT_GRAY_F64);
        dst.setStep(8);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }

}

void MatOp::RGBA2GRAY(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_RGBA_F32 && src.getMatType() != MAT_RGBA_U8  && src.getMatType() != MAT_RGBA_F64)
    {
        throw Exception(1,"[MatOp]: RGBA2GRAY src needs 4 channels! \n", __FILE__, __LINE__, __FUNCTION__);
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
    else if(src.getMatType() == MAT_RGBA_F64)
    {
        const double B = 0.114f;
        const double G = 0.587f;
        const double R = 1.f - B - G;

        dstData.u8 = new uint8_t[width*height*8]();
        double* srcF64 = src.getData().f64;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = i*width+j;
                dstData.f64[pos] = (R*srcF64[4*pos+0]+G*srcF64[4*pos+1]+B*srcF64[4*pos+2]);
            }
        }
        dst.release();
        dst.setChannel(1);
        dst.setMatType(MAT_GRAY_F64);
        dst.setStep(8);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

void MatOp::GRAY2RGB(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_GRAY_F32 && src.getMatType() != MAT_GRAY_U8 && src.getMatType() != MAT_GRAY_F64)
    {
        throw Exception(1,"[MatOp]: GRAY2RGB src needs 1 channel! \n", __FILE__, __LINE__, __FUNCTION__);
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
    else if(src.getMatType() == MAT_GRAY_F64)
    {
        dstData.u8 = new uint8_t[width*height*24]();
        double* srcF64 = src.getData().f64;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.f64[pos*3+0] = srcF64[pos];
                dstData.f64[pos*3+1] = srcF64[pos];
                dstData.f64[pos*3+2] = srcF64[pos];
            }
        }
        dst.release();
        dst.setChannel(3);
        dst.setMatType(MAT_RGB_F64);
        dst.setStep(24);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

void MatOp::GRAY2RGBA(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_GRAY_F32 && src.getMatType() != MAT_GRAY_U8 && src.getMatType() != MAT_GRAY_F64)
    {
        throw Exception(1,"[MatOp]: GRAY2RGBA src needs 1 channel! \n", __FILE__, __LINE__, __FUNCTION__);
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
    else if(src.getMatType() == MAT_GRAY_F64)
    {
        dstData.u8 = new uint8_t[width*height*32]();
        double* srcF64 = src.getData().f64;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.f64[pos*4+0] = srcF64[pos];
                dstData.f64[pos*4+1] = srcF64[pos];
                dstData.f64[pos*4+2] = srcF64[pos];
                dstData.f64[pos*4+3] = 1.f;
            }
        }
        dst.release();
        dst.setChannel(4);
        dst.setMatType(MAT_RGBA_F64);
        dst.setStep(32);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

void MatOp::RGB2RGBA(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_RGB_F32 && src.getMatType() != MAT_RGB_U8 && src.getMatType() != MAT_RGB_F64)
    {
        throw Exception(1,"[MatOp]: RGB2RGBA src needs 3 channels! \n", __FILE__, __LINE__, __FUNCTION__);
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
    else if(src.getMatType() == MAT_RGB_F64)
    {
        dstData.u8 = new uint8_t[width*height*32]();
        double* srcF64 = src.getData().f64;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.f64[pos*4+0] = srcF64[3*pos+0];
                dstData.f64[pos*4+1] = srcF64[3*pos+1];
                dstData.f64[pos*4+2] = srcF64[3*pos+2];
                dstData.f64[pos*4+3] = 1.f;
            }
        }
        dst.release();
        dst.setChannel(4);
        dst.setMatType(MAT_RGBA_F64);
        dst.setStep(32);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

void MatOp::RGBA2RGB(Mat &src, Mat &dst)
{
    if(src.getMatType() != MAT_RGBA_F32 && src.getMatType() != MAT_RGBA_U8 && src.getMatType() != MAT_RGBA_F64)
    {
        throw Exception(1,"[MatOp]: RGBA2RGB src needs 4 channels! \n", __FILE__, __LINE__, __FUNCTION__);
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
    else if(src.getMatType() == MAT_RGBA_F64)
    {
        dstData.u8 = new uint8_t[width*height*24]();
        double* srcF64 = src.getData().f64;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int pos = (i*width+j);
                dstData.f64[pos*3+0] = srcF64[pos*4+0];
                dstData.f64[pos*3+1] = srcF64[pos*4+1];
                dstData.f64[pos*3+2] = srcF64[pos*4+2];
            }
        }
        dst.release();
        dst.setChannel(3);
        dst.setMatType(MAT_RGB_F64);
        dst.setStep(24);
        dst.setWidth(width);
        dst.setHeight(height);
        dst.setU8Ptr(dstData.u8);
    }
}

void MatOp::flipV(Mat &mat)
{
    unsigned int line = mat.getWidth()*mat.getChannel();
#ifndef USE_OMP
    uint8_t* tmpLine = new uint8_t[line]();
#endif

#ifdef USE_OMP
    std::vector<uint8_t*> tmpData;
    for(int i=0;i<OMP_THREAD;i++)
    {
        tmpData.push_back(new uint8_t[line]());
    }
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for(int i=0; i<mat.getHeight()/2;i++)
    {
#ifdef USE_OMP
        uint8_t* tmpLine = tmpData[omp_get_thread_num()];
#endif
        memcpy(tmpLine,mat.getData().u8 + line*i, line);
        memcpy(mat.getData().u8+line*i, mat.getData().u8+line*(mat.getHeight()-i-1),line);
        memcpy(mat.getData().u8+line*(mat.getHeight()-i-1),tmpLine, line);
    }
#ifdef USE_OMP
    for(int i=0;i<OMP_THREAD;i++)
    {
        delete [] tmpData[i];
        tmpData[i] = nullptr;
    }
#endif
}

void MatOp::flipH(Mat &mat)
{
    if(mat.isU8Mat())
    {
        int n = mat.getChannel();
        uint32_t line   = mat.getWidth()*n;
        uint32_t width  = mat.getWidth();

#ifndef USE_OMP
        uint8_t* tmp = new uint8_t[n]();
#endif

#ifdef USE_OMP
        std::vector<uint8_t*> tmpData;
        for(int i=0;i<OMP_THREAD;i++)
        {
            tmpData.push_back(new uint8_t[n]());
        }
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i =0; i< mat.getHeight();i++)
        {
            for(int j=0; j<mat.getWidth()/2;j++)
            {
#ifdef USE_OMP
                uint8_t* tmp = tmpData[omp_get_thread_num()];
#endif
                memcpy(tmp,(mat.getData().u8+i*line + j*n),n);
                memcpy((mat.getData().u8+i*line + j*n),(mat.getData().u8+i*line + (width-j-1)*n),n);
                memcpy((mat.getData().u8+i*line + (width-j-1)*n),tmp,n);
            }
        }

#ifdef USE_OMP
        for(int i=0;i<OMP_THREAD;i++)
        {
            delete[] tmpData[i];
            tmpData[i] = nullptr;
        }
#endif

    }
    else if(mat.isF32Mat())
    {
        int n = mat.getChannel();
        uint32_t line   = mat.getWidth()*n;
        uint32_t width  = mat.getWidth();

#ifndef USE_OMP
        float* tmp = new float[n]();
#endif

#ifdef USE_OMP
        std::vector<float*> tmpData;
        for(int i=0;i<OMP_THREAD;i++)
        {
            tmpData.push_back(new float[n]());
        }
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i =0; i< mat.getHeight();i++)
        {
            for(int j=0; j<mat.getWidth()/2;j++)
            {
#ifdef USE_OMP
                float* tmp = tmpData[omp_get_thread_num()];
#endif
                memcpy(tmp,(mat.getData().f32+i*line + j*n),n);
                memcpy((mat.getData().f32+i*line + j*n),(mat.getData().f32+i*line + (width-j-1)*n),n);
                memcpy((mat.getData().f32+i*line + (width-j-1)*n),tmp,n);
            }
        }

#ifdef USE_OMP
        for(int i=0;i<OMP_THREAD;i++)
        {
            delete[] tmpData[i];
            tmpData[i] = nullptr;
        }
#endif
    }
    else if(mat.isF64Mat())
    {
        int n = mat.getChannel();
        uint32_t line   = mat.getWidth()*n;
        uint32_t width  = mat.getWidth();

#ifndef USE_OMP
        double* tmp = new double[n]();
#endif

#ifdef USE_OMP
        std::vector<double*> tmpData;
        for(int i=0;i<OMP_THREAD;i++)
        {
            tmpData.push_back(new double[n]());
        }
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for(int i =0; i< mat.getHeight();i++)
        {
            for(int j=0; j<mat.getWidth()/2;j++)
            {
#ifdef USE_OMP
                double* tmp = tmpData[omp_get_thread_num()];
#endif
                memcpy(tmp,(mat.getData().f64+i*line + j*n),n);
                memcpy((mat.getData().f64+i*line + j*n),(mat.getData().f64+i*line + (width-j-1)*n),n);
                memcpy((mat.getData().f64+i*line + (width-j)*n-1),tmp,n);
            }
        }

#ifdef USE_OMP
        for(int i=0;i<OMP_THREAD;i++)
        {
            delete[] tmpData[i];
            tmpData[i] = nullptr;
        }
#endif
    }
}

}
