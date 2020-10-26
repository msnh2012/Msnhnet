
#include "Msnhnet/cv/MsnhCVMat.h"

#define  STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define  STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "Msnhnet/core/MsnhGemm.h"

namespace Msnhnet
{

Mat::Mat()
{
}

Mat::Mat(const Mat &mat)
{

    this->_channel  = mat._channel;
    this->_height   = mat._height;
    this->_width    = mat._width;
    this->_matType  = mat._matType;
    this->_step     = mat._step;

    this->_data.u8  = new uint8_t[this->_width*this->_height*this->_step]();

    if(mat.getData().u8!=nullptr)
    {
        memcpy(this->_data.u8, mat.getData().u8, this->_width*this->_height*this->_step);
    }
}

Mat::Mat(const std::string &path)
{
    readImage(path);
}

Mat::~Mat()
{
    release();
}

Mat::Mat(const int &width, const int &height, const MatType &matType)
{
    this->_width    = width;
    this->_height   = height;
    this->_matType  = matType;

    switch (matType)
    {
    case MatType::MAT_GRAY_U8:
        this->_channel  = 1;
        this->_step     = 1;
        break;
    case MatType::MAT_GRAY_F32:
        this->_channel  = 1;
        this->_step     = 4;
        break;
    case MatType::MAT_GRAY_F64:
        this->_channel  = 1;
        this->_step     = 8;
        break;
    case MatType::MAT_RGB_U8:
        this->_channel  = 3;
        this->_step     = 3;
        break;
    case MatType::MAT_RGB_F32:
        this->_channel  = 3;
        this->_step     = 12;
        break;
    case MatType::MAT_RGB_F64:
        this->_channel  = 3;
        this->_step     = 24;
        break;
    case MatType::MAT_RGBA_U8:
        this->_channel  = 4;
        this->_step     = 4;
        break;
    case MatType::MAT_RGBA_F32:
        this->_channel  = 4;
        this->_step     = 16;
    case MatType::MAT_RGBA_F64:
        this->_channel  = 4;
        this->_step     = 32;
        break;
    }

    this->_data.u8      = new uint8_t[this->_width*this->_height*this->_step]();
}

Mat::Mat(const int &width, const int &height, const MatType &matType, void *data)
{
    this->_width    = width;
    this->_height   = height;
    this->_matType  = matType;

    switch (matType)
    {
    case MatType::MAT_GRAY_U8:
        this->_channel  = 1;
        this->_step     = 1;
        break;
    case MatType::MAT_GRAY_F32:
        this->_channel  = 1;
        this->_step     = 4;
        break;
    case MatType::MAT_GRAY_F64:
        this->_channel  = 1;
        this->_step     = 8;
        break;
    case MatType::MAT_RGB_U8:
        this->_channel  = 3;
        this->_step     = 3;
        break;
    case MatType::MAT_RGB_F32:
        this->_channel  = 3;
        this->_step     = 12;
        break;
    case MatType::MAT_RGB_F64:
        this->_channel  = 3;
        this->_step     = 24;
        break;
    case MatType::MAT_RGBA_U8:
        this->_channel  = 4;
        this->_step     = 4;
        break;
    case MatType::MAT_RGBA_F32:
        this->_channel  = 4;
        this->_step     = 16;
    case MatType::MAT_RGBA_F64:
        this->_channel  = 4;
        this->_step     = 32;
        break;
    }

    this->_data.u8 = new uint8_t[this->_width*this->_height*this->_step]();

    if(data!=nullptr)
    {
        memcpy(this->_data.u8, data, this->_width*this->_height*this->_step);
    }
}

void Mat::checkPixelType(const int &array, const int &fmt)
{
    if(fmt!='b' && fmt!='f' && fmt!='d' )
    {
        throw Exception(1,"[CV]: data type is not supported", __FILE__, __LINE__, __FUNCTION__);
    }

    switch (this->_matType)
    {
    case MatType::MAT_GRAY_U8:
        if(fmt!='b'||array!=1)
        {
            throw Exception(1,"[CV]: pixel type must be uint8_t", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_GRAY_F32:
        if(fmt!='f'||array!=1)
        {
            throw Exception(1,"[CV]: pixel type must be float", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_GRAY_F64:
        if(fmt!='d'||array!=1)
        {
            throw Exception(1,"[CV]: pixel type must be double", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGB_U8:
        if(fmt!='b'||array!=3)
        {
            throw Exception(1,"[CV]: pixel type must be Vec3U8", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGB_F32:
        if(fmt!='f'||array!=3)
        {
            throw Exception(1,"[CV]: pixel type must be Vec3F32", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGB_F64:
        if(fmt!='d'||array!=3)
        {
            throw Exception(1,"[CV]: pixel type must be Vec3F64", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGBA_U8:
        if(fmt!='b'||array!=4)
        {
            throw Exception(1,"[CV]: pixel type must be Vec4U8", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGBA_F32:
        if(fmt!='f'||array!=4)
        {
            throw Exception(1,"[CV]: pixel type must be Vec4F32", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGBA_F64:
        if(fmt!='f'||array!=4)
        {
            throw Exception(1,"[CV]: pixel type must be Vec4F64", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    }
}

void Mat::readImage(const std::string &path)
{
    release();
    this->_data.u8 = stbi_load(path.data(), &this->_width, &this->_height, &this->_channel, 0);

    if(this->_data.u8==nullptr)
    {
        throw Exception(1,"[CV]: img empty, maybe path error!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(this->_channel == 1)
    {
        this->_matType  = MatType::MAT_GRAY_U8;
        this->_step     = this->_channel;
    }
    else if(this->_channel == 3)
    {
        this->_matType  = MatType::MAT_RGB_U8;
        this->_step     = this->_channel;
    }
    else if(this->_channel == 4)
    {
        this->_matType  = MatType::MAT_RGBA_U8;
        this->_step     = this->_channel;
    }
}

void Mat::saveImage(const std::string &path, const SaveImageType &saveImageType, const int &quality)
{
    if(this->isEmpty())
    {
        throw Exception(1,"[CV]: img empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    float* f32Val  = nullptr;
    uint8_t* u8Val = nullptr;

    Mat tmpMat;

    if(saveImageType == SaveImageType::MAT_SAVE_HDR)
    {
        if(this->isF32Mat())
        {
            f32Val = this->_data.f32;
        }
        else
        {
            this->convertTo(tmpMat, CVT_DATA_TO_F32);
            f32Val = tmpMat.getData().f32;
        }
    }
    else
    {
        if(this->isU8Mat())
        {
            u8Val = this->_data.u8;
        }
        else
        {
            this->convertTo(tmpMat, CVT_DATA_TO_U8);
            u8Val = tmpMat.getData().u8;
        }
    }

    int ret;
    switch (saveImageType)
    {
    case SaveImageType::MAT_SAVE_BMP:
        ret = stbi_write_bmp(path.c_str(), this->_width, this->_height,this->_channel,u8Val);
        break;
    case SaveImageType::MAT_SAVE_JPG:
        ret = stbi_write_jpg(path.c_str(), this->_width, this->_height,this->_channel,u8Val, quality);
        break;
    case SaveImageType::MAT_SAVE_PNG:
        ret = stbi_write_png(path.c_str(), this->_width, this->_height,this->_channel,u8Val,0);
        break;
    case SaveImageType::MAT_SAVE_HDR:
        ret = stbi_write_hdr(path.c_str(), this->_width, this->_height,this->_channel,f32Val);
        break;
    case SaveImageType::MAT_SAVE_TGA:
        ret = stbi_write_tga(path.c_str(), this->_width, this->_height,this->_channel,u8Val);
        break;
    }

    if(ret<1)
    {
        throw Exception(1,"[CV]: save image error!", __FILE__, __LINE__, __FUNCTION__);
    }

}

void Mat::saveImage(const std::string &path, const int &quality)
{
    if(this->isEmpty())
    {
        throw Exception(1,"[CV]: img empty!", __FILE__, __LINE__, __FUNCTION__);
    }
    std::vector<std::string> splits;
    std::string tmpPath = path;
    ExString::split(splits, tmpPath, ".");
    std::string imgType = splits[splits.size()-1];
    if(imgType == "jpg" || imgType == "jpeg" || imgType == "JPG" || imgType == "JPEG")
    {
        saveImage(path,SaveImageType::MAT_SAVE_JPG,quality);
    }
    else if(imgType == "png" || imgType == "PNG")
    {
        saveImage(path,SaveImageType::MAT_SAVE_PNG,quality);
    }
    else if(imgType == "bmp" || imgType == "BMP")
    {
        saveImage(path,SaveImageType::MAT_SAVE_BMP,quality);
    }
    else if(imgType == "tga" || imgType == "TGA")
    {
        saveImage(path,SaveImageType::MAT_SAVE_TGA,quality);
    }
    else if(imgType == "hdr" || imgType == "HDR")
    {
        saveImage(path,SaveImageType::MAT_SAVE_TGA,quality);
    }
    else
    {
        throw Exception(1,"[CV]: unknown image type : "  + imgType, __FILE__, __LINE__, __FUNCTION__);
    }
}

void Mat::release()
{
    if(this->_data.u8!=nullptr)
    {
        delete[] this->_data.u8;
        this->_data.u8 = nullptr;
    }
    this->_width    = 0;
    this->_height   = 0;
    this->_channel  = 0;
    this->_step     = 0;
    this->_matType  = MatType::MAT_RGB_U8;
}

Mat &Mat::operator = (const Mat &mat) 

{
    if(this!=&mat)
    {
        release();
        this->_channel  = mat._channel;
        this->_width    = mat._width;
        this->_height   = mat._height;
        this->_step     = mat._step;
        this->_matType  = mat._matType;

        if(mat._data.u8!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat._data.u8, this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
    }
    return *this;
}

void Mat::copyTo(Mat &mat)
{

    mat.release();
    mat.setWidth(this->_width);
    mat.setHeight(this->_height);
    mat.setChannel(this->_channel);
    mat.setStep(this->_step);
    mat.setMatType(this->_matType);

    uint8_t* u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();

    if(this->_data.u8!=nullptr)
    {
        memcpy(u8Ptr, this->_data.u8, this->_width*this->_height*this->_step);
    }

    mat.setU8Ptr(u8Ptr);
}

void Mat::convertTo(Mat &dst, const CvtDataType &cvtDataType)
{
    if(this->isEmpty())
    {
        throw Exception(1,"[CV]: img empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    switch (cvtDataType)
    {
    case CVT_DATA_TO_F32:
        if(this->isF32Mat())
        {
            dst = *this;
        }
        else if(this->isF64Mat())
        {
            MatData data;
            data.u8 = new uint8_t[this->_width*this->_height*this->_channel*4]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        float val = static_cast<float>(this->_data.f64[i*this->_width*this->_channel + j*this->_channel + c]);
                        data.f32[i*this->_width*this->_channel + j*this->_channel + c] = val;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_F64)
            {
                dstType = MAT_GRAY_F32;
            }
            else if(this->_matType == MAT_RGB_F64)
            {
                dstType = MAT_RGB_F32;
            }
            else if(this->_matType == MAT_RGBA_F64)
            {
                dstType = MAT_RGBA_F32;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step/2;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(data.u8);
        }
        else if(this->isU8Mat())
        {
            MatData data;
            data.u8 = new uint8_t[this->_width*this->_height*this->_channel*4]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        float val = 1.0f*this->_data.u8[i*this->_width*this->_channel + j*this->_channel + c]/255;
                        data.f32[i*this->_width*this->_channel + j*this->_channel + c] = val;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_U8)
            {
                dstType = MAT_GRAY_F32;
            }
            else if(this->_matType == MAT_RGB_U8)
            {
                dstType = MAT_RGB_F32;
            }
            else if(this->_matType == MAT_RGBA_U8)
            {
                dstType = MAT_RGBA_F32;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step*4;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(data.u8);
        }
        break;
    case CVT_DATA_TO_F32_DIRECTLY:
        if(this->isF32Mat())
        {
            dst = *this;
        }
        else if(this->isF64Mat())
        {
            MatData data;
            data.u8 = new uint8_t[this->_width*this->_height*this->_channel*4]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        float val = static_cast<float>(this->_data.f64[i*this->_width*this->_channel + j*this->_channel + c]);
                        data.f32[i*this->_width*this->_channel + j*this->_channel + c] = val;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_F64)
            {
                dstType = MAT_GRAY_F32;
            }
            else if(this->_matType == MAT_RGB_F64)
            {
                dstType = MAT_RGB_F32;
            }
            else if(this->_matType == MAT_RGBA_F64)
            {
                dstType = MAT_RGBA_F32;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step/2;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(data.u8);
        }
        else if(this->isU8Mat())
        {
            MatData data;
            data.u8 = new uint8_t[this->_width*this->_height*this->_channel*4]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        float val = static_cast<float>(this->_data.u8[i*this->_width*this->_channel + j*this->_channel + c]);
                        data.f32[i*this->_width*this->_channel + j*this->_channel + c] = val;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_U8)
            {
                dstType = MAT_GRAY_F32;
            }
            else if(this->_matType == MAT_RGB_U8)
            {
                dstType = MAT_RGB_F32;
            }
            else if(this->_matType == MAT_RGBA_U8)
            {
                dstType = MAT_RGBA_F32;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step*4;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(data.u8);
        }
        break;
    case CVT_DATA_TO_U8:
        if(this->isU8Mat())
        {
            dst = *this;
        }
        else if(this->isF32Mat())
        {
            uint8_t* u8Ptr = new uint8_t[this->_width*this->_height*this->_channel]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        int val = static_cast<int>(this->_data.f32[i*this->_width*this->_channel + j*this->_channel + c]*255);
                        uint8_t finalVal = val>255?255:val;

                        u8Ptr[i*this->_width*this->_channel + j*this->_channel + c] = finalVal;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_F32 )
            {
                dstType = MAT_GRAY_U8;
            }
            else if(this->_matType == MAT_RGB_F32)
            {
                dstType = MAT_RGB_U8 ;
            }
            else if(this->_matType == MAT_RGBA_F32)
            {
                dstType = MAT_RGBA_U8 ;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step/4;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(u8Ptr);
        }
        else if(this->isF64Mat())
        {
            uint8_t* u8Ptr = new uint8_t[this->_width*this->_height*this->_channel]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        int val = static_cast<int>(this->_data.f64[i*this->_width*this->_channel + j*this->_channel + c]*255);
                        uint8_t finalVal = val>255?255:val;

                        u8Ptr[i*this->_width*this->_channel + j*this->_channel + c] = finalVal;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_F64 )
            {
                dstType = MAT_GRAY_U8;
            }
            else if(this->_matType == MAT_RGB_F64)
            {
                dstType = MAT_RGB_U8 ;
            }
            else if(this->_matType == MAT_RGBA_F64)
            {
                dstType = MAT_RGBA_U8 ;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step/8;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(u8Ptr);
        }
        break;
    case CVT_DATA_TO_U8_DIRECTLY:
        if(this->isU8Mat())
        {
            dst = *this;
        }
        else if(this->isF32Mat())
        {
            uint8_t* u8Ptr = new uint8_t[this->_width*this->_height*this->_channel]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        int val = static_cast<int>(this->_data.f32[i*this->_width*this->_channel + j*this->_channel + c]);
                        uint8_t finalVal = val>255?255:val;

                        u8Ptr[i*this->_width*this->_channel + j*this->_channel + c] = finalVal;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_F32 )
            {
                dstType = MAT_GRAY_U8;
            }
            else if(this->_matType == MAT_RGB_F32)
            {
                dstType = MAT_RGB_U8 ;
            }
            else if(this->_matType == MAT_RGBA_F32)
            {
                dstType = MAT_RGBA_U8 ;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step/4;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(u8Ptr);
        }
        else if(this->isF64Mat())
        {
            uint8_t* u8Ptr = new uint8_t[this->_width*this->_height*this->_channel]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        int val = static_cast<int>(this->_data.f64[i*this->_width*this->_channel + j*this->_channel + c]);
                        uint8_t finalVal = val>255?255:val;

                        u8Ptr[i*this->_width*this->_channel + j*this->_channel + c] = finalVal;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_F64 )
            {
                dstType = MAT_GRAY_U8;
            }
            else if(this->_matType == MAT_RGB_F64)
            {
                dstType = MAT_RGB_U8 ;
            }
            else if(this->_matType == MAT_RGBA_F64)
            {
                dstType = MAT_RGBA_U8 ;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step/8;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(u8Ptr);
        }
        break;
    case CVT_DATA_TO_F64:
        if(this->isF64Mat())
        {
            dst = *this;
        }
        else if(this->isF32Mat())
        {
            MatData data;
            data.u8 = new uint8_t[this->_width*this->_height*this->_channel*8]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        double val = static_cast<double>(this->_data.f32[i*this->_width*this->_channel + j*this->_channel + c]);
                        data.f64[i*this->_width*this->_channel + j*this->_channel + c] = val;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_F32 )
            {
                dstType = MAT_GRAY_F64;
            }
            else if(this->_matType == MAT_RGB_F32)
            {
                dstType = MAT_RGB_F64 ;
            }
            else if(this->_matType == MAT_RGBA_F32)
            {
                dstType = MAT_RGBA_F64 ;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step*2;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(data.u8);
        }
        else if(this->isU8Mat())
        {
            MatData data;
            data.u8 = new uint8_t[this->_width*this->_height*this->_channel*8]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        double val = 1.0*this->_data.u8[i*this->_width*this->_channel + j*this->_channel + c]/255;
                        data.f64[i*this->_width*this->_channel + j*this->_channel + c] = val;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_U8)
            {
                dstType = MAT_GRAY_F64;
            }
            else if(this->_matType == MAT_RGB_U8)
            {
                dstType = MAT_RGB_F64;
            }
            else if(this->_matType == MAT_RGBA_U8)
            {
                dstType = MAT_RGBA_F64;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step*8;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(data.u8);
        }
    case CVT_DATA_TO_F64_DIRECTLY:
        if(this->isF64Mat())
        {
            dst = *this;
        }
        else if(this->isF32Mat())
        {
            MatData data;
            data.u8 = new uint8_t[this->_width*this->_height*this->_channel*8]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        double val = static_cast<double>(this->_data.f32[i*this->_width*this->_channel + j*this->_channel + c]);
                        data.f64[i*this->_width*this->_channel + j*this->_channel + c] = val;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_F32 )
            {
                dstType = MAT_GRAY_F64;
            }
            else if(this->_matType == MAT_RGB_F32)
            {
                dstType = MAT_RGB_F64 ;
            }
            else if(this->_matType == MAT_RGBA_F32)
            {
                dstType = MAT_RGBA_F64 ;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step*2;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(data.u8);
        }
        else if(this->isU8Mat())
        {
            MatData data;
            data.u8 = new uint8_t[this->_width*this->_height*this->_channel*8]();
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
            for (int i = 0; i < this->_height; ++i)
            {
                for (int j = 0; j < this->_width; ++j)
                {
                    for (int c = 0; c < this->_channel; ++c)
                    {
                        double val = 1.0*this->_data.u8[i*this->_width*this->_channel + j*this->_channel + c];
                        data.f64[i*this->_width*this->_channel + j*this->_channel + c] = val;
                    }
                }
            }

            MatType dstType;

            if(this->_matType == MAT_GRAY_U8)
            {
                dstType = MAT_GRAY_F64;
            }
            else if(this->_matType == MAT_RGB_U8)
            {
                dstType = MAT_RGB_F64;
            }
            else if(this->_matType == MAT_RGBA_U8)
            {
                dstType = MAT_RGBA_F64;
            }

            int tmpCh   = this->_channel;
            int tmpStep = this->_step*8;
            int tmpW    = this->_width;
            int tmpH    = this->_height;

            dst.release();
            dst.setChannel(tmpCh);
            dst.setMatType(dstType);
            dst.setStep(tmpStep);
            dst.setWidth(tmpW);
            dst.setHeight(tmpH);
            dst.setU8Ptr(data.u8);
        }
    }
}

int Mat::getWidth() const
{
    return _width;
}

int Mat::getHeight() const
{
    return _height;
}

int Mat::getChannel() const
{
    return _channel;
}

int Mat::getStep() const
{
    return _step;
}

MatType Mat::getMatType() const
{
    return _matType;
}

MatData Mat::getData() const
{
    return _data;
}

void Mat::setWidth(int width)
{
    _width = width;
}

void Mat::setHeight(int height)
{
    _height = height;
}

void Mat::setChannel(int channel)
{
    _channel = channel;
}

void Mat::setStep(int step)
{
    _step = step;
}

void Mat::setMatType(const MatType &matType)
{
    _matType = matType;
}

void Mat::setU8Ptr(uint8_t * const &ptr)
{
    this->_data.u8 = ptr;
}

bool Mat::isEmpty() const
{
    if(this->_data.u8==nullptr||this->_height==0||this->_width==0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

Vec2I32 Mat::getSize()
{
    return Vec2I32(this->_width, this->_height);
}

size_t Mat::getDataNum()
{
    return this->_width*this->_height*this->_channel;
}

uint8_t Mat::getPerDataByteNum()
{
    return static_cast<uint8_t>(this->_step/this->_channel);
}

Mat Mat::eye(const int &num, const MatType &matType)
{
    if(matType != MAT_GRAY_U8 && matType != MAT_GRAY_F32 && matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[CV]: Only one channel is supported!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmp(num,num, matType);

    if(matType == MAT_GRAY_U8)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().u8[i*tmp.getWidth()+i] = 1;
        }
    }
    else if(matType == MAT_GRAY_F32)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().f32[i*tmp.getWidth()+i] = 1.f;
        }
    }
    else if(matType == MAT_GRAY_F64)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().f64[i*tmp.getWidth()+i] = 1.0;
        }
    }

    return tmp;

}

Mat Mat::dense(const int &width, const int &height, const MatType &matType, const float &val)
{
    if(matType != MAT_GRAY_U8 && matType != MAT_GRAY_F32 && matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[CV]: Only one channel is supported!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmp(width,height, matType);

    if(matType == MAT_GRAY_U8)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            for (int j = 0; j < tmp.getWidth(); ++j)
            {
                tmp.getData().u8[i*tmp.getWidth()+j] = static_cast<uint8_t>(val) ;
            }
        }
    }
    else if(matType == MAT_GRAY_F32)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            for (int j = 0; j < tmp.getWidth(); ++j)
            {
                tmp.getData().f32[i*tmp.getWidth()+j] = val;
            }
        }
    }
    else if(matType == MAT_GRAY_F64)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            for (int j = 0; j < tmp.getWidth(); ++j)
            {
                tmp.getData().f64[i*tmp.getWidth()+j] = val;
            }
        }
    }
    return tmp;
}

Mat Mat::diag(const int &num, const MatType &matType, const float &val)
{
    if(matType != MAT_GRAY_U8 && matType != MAT_GRAY_F32 && matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[CV]: Only one channel is supported!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmp(num,num, matType);

    if(matType == MAT_GRAY_U8)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().u8[i*tmp.getWidth()+i] = static_cast<uint8_t>(val) ;
        }
    }
    else if(matType == MAT_GRAY_F32)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().f32[i*tmp.getWidth()+i] = val;
        }
    }
    else if(matType == MAT_GRAY_F64)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().f64[i*tmp.getWidth()+i] = val;
        }
    }

    return tmp;
}

Mat Mat::random(const int &width, const int &height, const MatType &matType)
{
    if(matType != MAT_GRAY_U8 && matType != MAT_GRAY_F32 && matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[CV]: Only one channel is supported!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmp(width,height, matType);

    srand( (unsigned)time( nullptr ) );

    if(matType == MAT_GRAY_U8)
    {

        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            for (int j = 0; j < tmp.getWidth(); ++j)
            {
                tmp.getData().u8[i*tmp.getWidth()+j] = static_cast<uint8_t>(rand()%255);
            }
        }
    }
    else if(matType == MAT_GRAY_F32)
    {
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            for (int j = 0; j < tmp.getWidth(); ++j)
            {
                tmp.getData().f32[i*tmp.getWidth()+j] = randUniform(0.f,1.f);
            }
        }
    }
    else if(matType == MAT_GRAY_F64)
    {
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            for (int j = 0; j < tmp.getWidth(); ++j)
            {
                tmp.getData().f64[i*tmp.getWidth()+j] = randUniform(0.0,1.0);
            }
        }
    }
    return tmp;
}

Mat Mat::randomDiag(const int &num, const MatType &matType)
{
    if(matType != MAT_GRAY_U8 && matType != MAT_GRAY_F32 && matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[CV]: Only one channel is supported!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmp(num,num, matType);

    srand( (unsigned)time( nullptr ) );

    if(matType == MAT_GRAY_U8)
    {
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().u8[i*tmp.getWidth()+i] = static_cast<uint8_t>(rand()%255);
        }
    }
    else if(matType == MAT_GRAY_F32)
    {
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().f32[i*tmp.getWidth()+i] = randUniform(0.f,1.f);
        }
    }
    else if(matType == MAT_GRAY_F64)
    {
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().f64[i*tmp.getWidth()+i] = randUniform(0.0,1.0);
        }
    }
    return tmp;
}

Mat Mat::transpose()
{
    if(this->isEmpty())
    {
        throw Exception(1,"[CV]: Mat empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(!this->isOneChannel())
    {
        throw Exception(1,"[CV]: Only one channel is supported!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat(this->_height,this->_width,this->_matType);

    if(isU8Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < this->_height; ++i)
        {
            for (int j = 0; j < this->_width; ++j)
            {
                tmpMat.getData().u8[j*tmpMat.getWidth()+i] = this->_data.u8[i*this->_width + j];
            }
        }
    }
    else if(isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < this->_height; ++i)
        {
            for (int j = 0; j < this->_width; ++j)
            {
                tmpMat.getData().f32[j*tmpMat.getWidth()+i] = this->_data.f32[i*this->_width + j];
            }
        }
    }
    else if(isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < this->_height; ++i)
        {
            for (int j = 0; j < this->_width; ++j)
            {
                tmpMat.getData().f64[j*tmpMat.getWidth()+i] = this->_data.f64[i*this->_width + j];
            }
        }
    }
    return tmpMat;
}

double Mat::det()
{
    if(this->_matType != MAT_GRAY_F32 && this->_matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[CV]: Only one channel F32 and F64 Mat is supported!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(this->_height!=this->_width)
    {
        throw Exception(1,"[CV]: Height must equal width!", __FILE__, __LINE__, __FUNCTION__);
    }

    double val = 0;

    if(this->_matType == MAT_GRAY_F32)
    {
        if(this->_height == 2)
        {
            val = this->_data.f32[0]*this->_data.f32[3] - this->_data.f32[1]*this->_data.f32[2];
        }
        else if(this->_height == 3)
        {
            val = this->_data.f32[0]*this->_data.f32[4]*this->_data.f32[8] + this->_data.f32[1]*this->_data.f32[5]*this->_data.f32[6] + this->_data.f32[2]*this->_data.f32[3]*this->_data.f32[7]
                    -this->_data.f32[0]*this->_data.f32[7]*this->_data.f32[5] - this->_data.f32[3]*this->_data.f32[1]*this->_data.f32[8] - this->_data.f32[2]*this->_data.f32[4]*this->_data.f32[6];
        }
        else
        {
            Mat tmpMat = *this;

            int k = 0;
            int m = tmpMat.getWidth();
            int p = 1;

            for(int i = 0; i < m; i++ )
            {
                k = i;

                for(int j = i+1; j < m; j++ )
                {

                    if( std::abs(tmpMat.getData().f32[j*m + i]) > std::abs(tmpMat.getData().f32[k*m + i]))
                    {
                        k = j;
                    }
                }

                if( std::abs(tmpMat.getData().f32[k*m + i]) < std::numeric_limits<float>::epsilon() )
                    return 0;

                if( k != i )
                {
                    for(int j = i; j < m; j++ )
                    {
                        std::swap(tmpMat.getData().f32[i*m + j], tmpMat.getData().f32[k*m + j]);
                    }
                    p = -p;
                }

                float d = -1/tmpMat.getData().f32[i*m + i];

                for(int j = i+1; j < m; j++ )
                {
                    float alpha = tmpMat.getData().f32[j*m + i]*d;

                    for( k = i; k < m; k++ )
                    {
                        tmpMat.getData().f32[j*m + k] += alpha*tmpMat.getData().f32[i*m + k];
                    }
                }
            }

            val = 1;

            for (int i = 0; i < m; ++i)
            {
                val *= tmpMat.getData().f32[i*m+i];
            }
            val = val*p;

        }
    }
    else
    {
        if(this->_height == 2)
        {
            val = this->_data.f64[0]*this->_data.f64[3] - this->_data.f64[1]*this->_data.f64[2];
        }
        else if(this->_height == 3)
        {
            val = this->_data.f64[0]*this->_data.f64[4]*this->_data.f64[8] + this->_data.f64[1]*this->_data.f64[5]*this->_data.f64[6] + this->_data.f64[2]*this->_data.f64[3]*this->_data.f64[7]
                    -this->_data.f64[0]*this->_data.f64[7]*this->_data.f64[5] - this->_data.f64[3]*this->_data.f64[1]*this->_data.f64[8] - this->_data.f64[2]*this->_data.f64[4]*this->_data.f64[6];
        }
        else
        {
            Mat tmpMat = *this;

            int k = 0;
            int m = tmpMat.getWidth();
            int p = 1;

            for(int i = 0; i < m; i++ )
            {
                k = i;

                for(int j = i+1; j < m; j++ )
                {

                    if( std::abs(tmpMat.getData().f64[j*m + i]) > std::abs(tmpMat.getData().f64[k*m + i]))
                    {
                        k = j;
                    }
                }

                if( std::abs(tmpMat.getData().f64[k*m + i]) < std::numeric_limits<double>::epsilon() )
                    return 0;

                if( k != i )
                {
                    for(int j = i; j < m; j++ )
                    {
                        std::swap(tmpMat.getData().f64[i*m + j], tmpMat.getData().f64[k*m + j]);
                    }
                    p = -p;
                }

                double d = -1/tmpMat.getData().f64[i*m + i];

                for(int j = i+1; j < m; j++ )
                {
                    double alpha = tmpMat.getData().f64[j*m + i]*d;

                    for( k = i; k < m; k++ )
                    {
                        tmpMat.getData().f64[j*m + k] += alpha*tmpMat.getData().f64[i*m + k];
                    }
                }
            }

            val = 1;

            for (int i = 0; i < m; ++i)
            {
                val *= tmpMat.getData().f64[i*m+i];
            }
            val = val*p;

        }
    }

    return val;
}

Mat Mat::invert()
{
    if(this->_matType != MAT_GRAY_F32 && this->_matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[CV]: Only one channel F32/F64 Mat is supported!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(this->_height!=this->_width)
    {
        throw Exception(1,"[CV]: Height must equal width!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat A = *this;
    int m = A.getWidth();
    Mat B = Mat::eye(m,A.getMatType());
    int k = 0;

    if(this->_matType == MAT_GRAY_F32)
    {

        for(int i = 0; i < m; i++ )
        {
            k = i;

            for(int j = i+1; j < m; j++ )
            {

                if( std::abs(A.getData().f32[j*m + i]) > std::abs(A.getData().f32[k*m + i]))
                {
                    k = j;
                }
            }

            if( std::abs(A.getData().f32[k*m + i]) < std::numeric_limits<float>::epsilon() )
                throw Exception(1,"[CV]: det=0, no invert mat!", __FILE__, __LINE__, __FUNCTION__);

            if( k != i )
            {
                for(int j = i; j < m; j++ )
                {
                    std::swap(A.getData().f32[i*m + j], A.getData().f32[k*m + j]);
                }

                for(int j = 0; j < m; j++ )
                {
                    std::swap(B.getData().f32[i*m + j], B.getData().f32[k*m + j]);
                }

            }

            float d = -1/A.getData().f32[i*m + i];

            for(int j = i+1; j < m; j++ )
            {
                float alpha = A.getData().f32[j*m + i]*d;

                for( k = i+1; k < m; k++ )
                {
                    A.getData().f32[j*m + k] += alpha*A.getData().f32[i*m + k];
                }

                for( k = 0; k < m; k++ )
                {
                    B.getData().f32[j*m + k] += alpha*B.getData().f32[i*m + k];
                }
            }

            A.getData().f32[i*m + i] = -d; 

        }

        for(int i = m-1; i >= 0; i-- )
        {
            for(int j = 0; j < m; j++ )
            {
                float s = B.getData().f32[i*m + j];
                for( k = i+1; k < m; k++ )
                {
                    s -= A.getData().f32[i*m + k]*B.getData().f32[k*m + j];
                }
                B.getData().f32[i*m + j] = s*A.getData().f32[i*m + i];
            }
        }
    }
    else
    {

        for(int i = 0; i < m; i++ )
        {
            k = i;

            for(int j = i+1; j < m; j++ )
            {

                if( std::abs(A.getData().f64[j*m + i]) > std::abs(A.getData().f64[k*m + i]))
                {
                    k = j;
                }
            }

            if( std::abs(A.getData().f64[k*m + i]) < std::numeric_limits<double>::epsilon() )
                throw Exception(1,"[CV]: det=0, no invert mat!", __FILE__, __LINE__, __FUNCTION__);

            if( k != i )
            {
                for(int j = i; j < m; j++ )
                {
                    std::swap(A.getData().f64[i*m + j], A.getData().f64[k*m + j]);
                }

                for(int j = 0; j < m; j++ )
                {
                    std::swap(B.getData().f64[i*m + j], B.getData().f64[k*m + j]);
                }

            }

            double d = -1/A.getData().f64[i*m + i];

            for(int j = i+1; j < m; j++ )
            {
                double alpha = A.getData().f64[j*m + i]*d;

                for( k = i+1; k < m; k++ )
                {
                    A.getData().f64[j*m + k] += alpha*A.getData().f64[i*m + k];
                }

                for( k = 0; k < m; k++ )
                {
                    B.getData().f64[j*m + k] += alpha*B.getData().f64[i*m + k];
                }
            }

            A.getData().f64[i*m + i] = -d; 

        }

        for(int i = m-1; i >= 0; i-- )
        {
            for(int j = 0; j < m; j++ )
            {
                double s = B.getData().f64[i*m + j];
                for( k = i+1; k < m; k++ )
                {
                    s -= A.getData().f64[i*m + k]*B.getData().f64[k*m + j];
                }
                B.getData().f64[i*m + j] = s*A.getData().f64[i*m + i];
            }
        }
    }

    return B;
}

void Mat::printMat()
{
    if(isF32Mat())
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            std::cout<<"{"<<std::endl;
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<19|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==19)
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                        }
                        else if(j<19 || j==(this->_width-1) )
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<std::setiosflags(std::ios::fixed)<<this->_data.f32[c*this->_width*this->_height + i*this->_width + j]<<" ";
                        }
                    }
                    std::cout<<std::endl;
                }
                else if(i == 20)
                {
                    std::cout<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                    std::cout<<std::endl;
                }

            }
            std::cout<<"},"<<std::endl;
        }
    }
    else if(isF64Mat())
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            std::cout<<"{"<<std::endl;
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<9|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==9)
                        {

                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                        }
                        else if(j<9 || j==(this->_width-1) )
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(12)<<std::setiosflags(std::ios::fixed)<<this->_data.f64[c*this->_width*this->_height + i*this->_width + j]<<" ";
                        }
                    }
                    std::cout<<std::endl;
                }
                else if(i == 10)
                {
                    std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                    std::cout<<std::endl;
                }

            }
            std::cout<<"},"<<std::endl;
        }
    }
    else
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            std::cout<<"{"<<std::endl;
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<19|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==19)
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(6)<<" ... ";
                        }
                        else if(j<19 || j==(this->_width-1) )
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(6)<<static_cast<int>(this->_data.u8[c*this->_width*this->_height + i*this->_width + j])<<std::setw(1)<<" ";
                        }
                    }
                    std::cout<<std::endl;
                }
                else if(i == 20)
                {
                    std::cout<<std::setiosflags(std::ios::left)<<std::setw(6)<<" ... ";
                    std::cout<<std::endl;
                }
            }
            std::cout<<"},"<<std::endl;
        }
    }

    std::cout<<"width: "<<this->_width<<" , height: "<<this->_height<<" , channels: "<<this->_channel<<std::endl;
}

bool Mat::isF32Mat() const
{
    return (this->_matType == MAT_GRAY_F32 || this->_matType == MAT_RGB_F32 || this->_matType == MAT_RGBA_F32);
}

bool Mat::isF64Mat() const
{
    return (this->_matType == MAT_GRAY_F64 || this->_matType == MAT_RGB_F64 || this->_matType == MAT_RGBA_F64);
}

bool Mat::isU8Mat() const
{
    return  (this->_matType == MAT_GRAY_U8 || this->_matType == MAT_RGB_U8 || this->_matType == MAT_RGBA_U8);
}

bool Mat::isOneChannel() const
{
    return (this->_matType == MAT_GRAY_U8 || this->_matType == MAT_GRAY_F32|| this->_matType == MAT_GRAY_F64);
}

Mat Mat::add(const Mat &A, const Mat &B)
{
    return A+B;
}

Mat Mat::sub(const Mat &A, const Mat &B)
{
    return A-B;
}

Mat Mat::dot(const Mat &A, const Mat &B)
{
    return A*B;
}

Mat Mat::div(const Mat &A, const Mat &B)
{
    return A/B;
}

Mat Mat::mul(const Mat &A, const Mat &B)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[CV]: Mat B is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(A.getWidth() != B.getHeight())
    {
        throw Exception(1,"[CV]: Mat A'W != B'H !", __FILE__, __LINE__, __FUNCTION__);
    }

    if(A.getMatType() != B.getMatType() || A.getChannel() != B.getChannel() || A.getStep() != B.getStep())
    {
        throw Exception(1,"[CV]: mat properties not equal!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(!A.isF32Mat() && !A.isF64Mat() && !A.isOneChannel())
    {
        throw Exception(1,"[CV]: mat must be f32/f64 1 channel mat!", __FILE__, __LINE__, __FUNCTION__);
    }

    SimdInfo simdInfo;

    if(A.isF32Mat())
    {

        Mat C(B.getWidth(),A.getHeight(),MatType::MAT_GRAY_F32);
        Gemm::cpuGemm(0,0,A.getHeight(),B.getWidth(),A.getWidth(),1,A.getData().f32,A.getWidth(),B.getData().f32,B.getWidth(),1,C.getData().f32,C.getWidth(), simdInfo.getSupportAVX2());
        return C;
    }
    else
    {
        Mat C(B.getWidth(),A.getHeight(),MatType::MAT_GRAY_F64);
        Gemm::cpuGemm(0,0,A.getHeight(),B.getWidth(),A.getWidth(),1,A.getData().f64,A.getWidth(),B.getData().f64,B.getWidth(),1,C.getData().f64,C.getWidth(), simdInfo.getSupportAVX2());
        return C;
    }
}

Mat operator +(const Mat &A, const Mat &B)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[CV]: Mat B is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(A._matType != B._matType || A._channel != B._channel || A._step != B._step ||
            A._width != B._width || A._height != B._height)
    {
        throw Exception(1,"[CV]: mat properties not equal!", __FILE__, __LINE__, __FUNCTION__);
    }

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int add = A._data.u8[i] + B._data.u8[i];

            add = (add>255)?255:add;

            tmpMat._data.u8[i] = static_cast<uint8_t>(add);

        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]+B._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i]+B._data.f64[i];
        }
    }
    return tmpMat;
}

Mat operator +(const double &a, const Mat &A)
{

    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
        int addVal = static_cast<int>(a);
        addVal = addVal>255?255:addVal;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int add = A._data.u8[i] + addVal;

            add = (add>255)?255:add;

            tmpMat._data.u8[i] = static_cast<uint8_t>(add);

        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]+ static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i]+ a;
        }
    }
    return tmpMat;
}

Mat operator +(const Mat &A, const double &a)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(tmpMat.isEmpty())
    {
        throw Exception(1,"[CV]: mat empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
        int addVal = static_cast<int>(a);
        addVal = addVal>255?255:addVal;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int add = A._data.u8[i] + addVal;

            add = (add>255)?255:add;

            tmpMat._data.u8[i] = static_cast<uint8_t>(add);

        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]+ static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i]+ a;
        }
    }
    return tmpMat;
}

Mat operator -(const Mat &A, const Mat &B)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[CV]: Mat B is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(A._matType != B._matType || A._channel != B._channel || A._step != B._step ||
            A._width != B._width || A._height != B._height)
    {
        throw Exception(1,"[CV]: mat properties not equal!", __FILE__, __LINE__, __FUNCTION__);
    }

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int sub = A._data.u8[i] - B._data.u8[i];

            sub = (sub<0)?0:sub;

            tmpMat._data.u8[i] = static_cast<uint8_t>(sub);
        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]-B._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i]-B._data.f64[i];
        }
    }
    return tmpMat;
}

Mat operator -(const double &a, const Mat &A)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(tmpMat.isEmpty())
    {
        throw Exception(1,"[CV]: mat empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
        int subVal = static_cast<int>(a);
        subVal = subVal>255?255:subVal;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int sub = subVal - A._data.u8[i];

            sub = (sub<0)?0:sub;

            tmpMat._data.u8[i] = static_cast<uint8_t>(sub);
        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = static_cast<float>(a) - A._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = a - A._data.f64[i];
        }
    }
    return tmpMat;
}

Mat operator -(const Mat &A, const double &a)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(tmpMat.isEmpty())
    {
        throw Exception(1,"[CV]: mat empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
        int subVal = static_cast<int>(a);
        subVal = subVal>255?255:subVal;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int sub = A._data.u8[i]-subVal;
            sub = (sub>255)?255:sub;
            sub = (sub<0)?0:sub;

            tmpMat._data.u8[i] = static_cast<uint8_t>(sub);
        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i] - static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i] - a;
        }
    }
    return tmpMat;
}

Mat operator *(const Mat &A, const Mat &B)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[CV]: Mat B is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(A._matType != B._matType || A._channel != B._channel || A._step != B._step ||
            A._width != B._width || A._height != B._height)
    {
        throw Exception(1,"[CV]: mat properties not equal!", __FILE__, __LINE__, __FUNCTION__);
    }

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int mul = A._data.u8[i] * B._data.u8[i];

            mul = (mul>255)?255:mul;

            tmpMat._data.u8[i] = static_cast<uint8_t>(mul);

        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i] * B._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i] * B._data.f64[i];
        }
    }
    return tmpMat;
}

Mat operator *(const double &a, const Mat &A)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
        int mulVal = static_cast<int>(a);
        mulVal = mulVal>255?255:mulVal;
        mulVal = mulVal<0?0:mulVal;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int mul = A._data.u8[i]* mulVal;

            mul = (mul>255)?255:mul;

            tmpMat._data.u8[i] = static_cast<uint8_t>(mul);

        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]*static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i]*a;
        }
    }
    return tmpMat;
}

Mat operator *(const Mat &A, const double &a)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
        int mulVal = static_cast<int>(a);
        mulVal = mulVal>255?255:mulVal;
        mulVal = mulVal<0?0:mulVal;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int mul = A._data.u8[i]*mulVal;

            mul = (mul>255)?255:mul;

            tmpMat._data.u8[i] = static_cast<uint8_t>(mul);

        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]*static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i]*a;
        }
    }
    return tmpMat;
}

Mat operator /(const Mat &A, const Mat &B)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[CV]: Mat B is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(A._matType != B._matType || A._channel != B._channel || A._step != B._step ||
            A._width != B._width || A._height != B._height)
    {
        throw Exception(1,"[CV]: mat properties not equal!", __FILE__, __LINE__, __FUNCTION__);
    }

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int div = 0;

            if(A._data.u8[i] == 0 || B._data.u8[i] == 0)
            {
                div = 0;
            }
            else
            {
                div = A._data.u8[i] / B._data.u8[i];
            }

            div = (div>255)?255:div;

            tmpMat._data.u8[i] = static_cast<uint8_t>(div);

        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i] / B._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i] / B._data.f64[i];
        }
    }
    return tmpMat;
}

Mat operator /(const double &a, const Mat &A)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
        int divVal = static_cast<int>(a);
        divVal = divVal<0?0:divVal;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int div =  0;

            if(divVal == 0 || A._data.u8[i] == 0)
            {
                div = 0;
            }
            else
            {
                div = static_cast<int>(divVal / A._data.u8[i]);
            }
            div = (div>255)?255:div;
            tmpMat._data.u8[i] = static_cast<uint8_t>(div);
        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = static_cast<float>(a) / A._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = a / A._data.f64[i];
        }
    }
    return tmpMat;
}

Mat operator /(const Mat &A, const double &a)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[CV]: Mat A is empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    int dataN = A._width*A._height;

    if(tmpMat.isU8Mat())
    {
        int divVal = static_cast<int>(a);
        divVal = divVal<0?0:divVal;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int div =  0;

            if(A._data.u8[i] == 0 || divVal == 0)
            {
                div = 0;
            }
            else
            {
                div = static_cast<int>(A._data.u8[i]/divVal);
            }
            div = (div>255)?255:div;
            tmpMat._data.u8[i] = static_cast<uint8_t>(div);
        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]/static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i]/a;
        }
    }
    return tmpMat;
}
}
