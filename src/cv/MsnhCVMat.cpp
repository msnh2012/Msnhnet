
#include "Msnhnet/cv/MsnhCVMat.h"

#define  STB_IMAGE_IMPLEMENTATION
#include "../3rdparty/stb/stb_image.h"

#define  STB_IMAGE_WRITE_IMPLEMENTATION
#include "../3rdparty/stb/stb_image_write.h"

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

#ifdef USE_R_VALUE_REF
Mat::Mat(Mat&& mat)
{

    this->_channel  = mat._channel;
    this->_height   = mat._height;
    this->_width    = mat._width;
    this->_matType  = mat._matType;
    this->_step     = mat._step;
    this->_data.u8  = mat._data.u8;
    mat.setDataNull();
}
#endif

Mat::Mat(const std::string &path)
{
    readImage(path);
}

Mat::~Mat()
{
    release();
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

void Mat::checkPixelType(const int &array, const int &fmt) const
{
    if(fmt!='b' && fmt!='f' && fmt!='d' )
    {
        throw Exception(1,"[Mat]: data type is not supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    switch (this->_matType)
    {
    case MatType::MAT_GRAY_U8:
        if(fmt!='b'||array!=1)
        {
            throw Exception(1,"[Mat]: pixel type must be uint8_t! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_GRAY_F32:
        if(fmt!='f'||array!=1)
        {
            throw Exception(1,"[Mat]: pixel type must be float! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_GRAY_F64:
        if(fmt!='d'||array!=1)
        {
            throw Exception(1,"[Mat]: pixel type must be double! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGB_U8:
        if(fmt!='b'||array!=3)
        {
            throw Exception(1,"[Mat]: pixel type must be Vec3U8! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGB_F32:
        if(fmt!='f'||array!=3)
        {
            throw Exception(1,"[Mat]: pixel type must be Vec3F32! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGB_F64:
        if(fmt!='d'||array!=3)
        {
            throw Exception(1,"[Mat]: pixel type must be Vec3F64! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGBA_U8:
        if(fmt!='b'||array!=4)
        {
            throw Exception(1,"[Mat]: pixel type must be Vec4U8! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGBA_F32:
        if(fmt!='f'||array!=4)
        {
            throw Exception(1,"[Mat]: pixel type must be Vec4F32! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        break;
    case MatType::MAT_RGBA_F64:
        if(fmt!='f'||array!=4)
        {
            throw Exception(1,"[Mat]: pixel type must be Vec4F64! \n", __FILE__, __LINE__, __FUNCTION__);
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
        throw Exception(1,"[Mat]: img empty, maybe path error! \n", __FILE__, __LINE__, __FUNCTION__);
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
        throw Exception(1,"[Mat]: img empty! \n", __FILE__, __LINE__, __FUNCTION__);
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
        ret = stbi_write_jpg(path.c_str(), this->_width, this->_height,this->_channel,u8Val,quality);
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
        throw Exception(1,"[Mat]: save image error! \n", __FILE__, __LINE__, __FUNCTION__);
    }

}

void Mat::saveImage(const std::string &path, const int &quality)
{
    if(this->isEmpty())
    {
        throw Exception(1,"[Mat]: img empty! \n", __FILE__, __LINE__, __FUNCTION__);
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
        throw Exception(1,"[Mat]: unknown image type : "  + imgType + "! \n", __FILE__, __LINE__, __FUNCTION__);
    }
}

Mat Mat::rowRange(int startCol, int cnts)
{
    if(startCol < 0 || cnts < 0 || startCol >= this->_height || (startCol + cnts) > this->_height)
    {
        throw Exception(1,"[Mat]: out of memory!\n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmp(this->_width,cnts,this->_matType);
    memcpy(tmp.getBytes(), this->_data.u8+ startCol*this->_width*this->_step, cnts*this->_width*this->_step);
    return tmp;
}

Mat Mat::colRange(int startRow, int cnts)
{
    if(startRow < 0 || cnts < 0 || startRow >= this->_width || (startRow + cnts) > this->_width)
    {
        throw Exception(1,"[Mat]: out of memory!\n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmp(cnts ,this->_height,this->_matType);

    std::cout<<this->getByteNum()<<std::endl;

    for (int i = 0; i < this->_height; ++i)
    {
        memcpy(tmp.getBytes() + i*cnts*this->_step, this->_data.u8 + (i*this->_width + startRow)*this->_step, cnts*this->_step);
    }

    return tmp;
}

std::vector<char> Mat::encodeToMemory(const MatEncodeType &encodeType, const int &jpgQuality)
{
    if(this->isEmpty())
    {
        throw Exception(1, "[Mat]: Mat empty! \n",__FILE__,__LINE__,__FUNCTION__);
    }
    Msnhnet::Mat mat = *this;

    if(!mat.isU8Mat())
    {
        mat.convertTo(mat, CVT_DATA_TO_U8);
    }

    std::vector<char> picData;
    if(encodeType== MAT_ENCODE_JPG)
    {
        stbi_write_jpg_to_func(bufferFromCallback,&picData, mat.getWidth(), mat.getHeight(), mat.getChannel() ,mat.getData().u8,jpgQuality);
    }
    else if(encodeType== MAT_ENCODE_PNG)
    {
        stbi_write_png_to_func(bufferFromCallback,&picData, mat.getWidth(), mat.getHeight(), mat.getChannel() ,mat.getData().u8,0);
    }

    return picData;
}

void Mat::decodeFromMemory(char *data, const size_t &dataLen)
{
    release();
    this->_data.u8 = stbi_load_from_memory(reinterpret_cast<stbi_uc*>(data), dataLen, &this->_width, &this->_height, &this->_channel,0);

    if(this->_data.u8==nullptr)
    {
        throw Exception(1,"[Mat]: Decode from memory error! \n", __FILE__, __LINE__, __FUNCTION__);
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

#ifdef USE_R_VALUE_REF
Mat &Mat::operator=(Mat &&mat)
{
    if(this!=&mat)
    {
        release();
        this->_channel  = mat._channel;
        this->_width    = mat._width;
        this->_height   = mat._height;
        this->_step     = mat._step;
        this->_matType  = mat._matType;
        this->_data.u8  = mat._data.u8;
        mat.setDataNull();
    }
    return *this;
}
#endif

Mat &Mat::operator +=(const Mat &A)
{
    *this = *this + A;
    return *this;
}

Mat &Mat::operator +=(const double &a)
{
    *this = *this + a;
    return *this;
}

Mat &Mat::operator -=(const Mat &A)
{
    *this = *this - A;
    return *this;
}

Mat &Mat::operator -=(const double &a)
{
    *this = *this - a;
    return *this;
}

Mat &Mat::operator *=(const Mat &A)
{
    *this = *this * A;
    return *this;
}

Mat &Mat::operator *=(const double &a)
{
    *this = *this * a;
    return *this;
}

Mat &Mat::operator /=(const Mat &A)
{
    *this = *this / A;
    return *this;
}

Mat &Mat::operator /=(const double &a)
{
    *this = *this / a;
    return *this;
}

bool operator!=(const Mat &A, const Mat &B)
{
    if(A.getMatType()!=B.getMatType() || A.getWidth()!=B.getWidth() || A.getHeight()!=B.getHeight() || A.getChannel()!=B.getChannel())
    {
        return true;
    }

    if(A.isU8Mat())
    {
        for (size_t i = 0; i < A.getDataNum(); ++i)
        {
            if(A.getData().u8[i] != B.getData().u8[i])
            {
                return true;
            }
        }
    }

    if(A.isF32Mat())
    {
        for (size_t i = 0; i < A.getDataNum(); ++i)
        {
            if(fabsf(A.getData().f32[i] - B.getData().f32[i])>MSNH_F32_EPS)
            {
                return true;
            }
        }
    }

    if(A.isF64Mat())
    {
        for (size_t i = 0; i < A.getDataNum(); ++i)
        {
            if(std::fabs(A.getData().f64[i] - B.getData().f64[i])>MSNH_F64_EPS)
            {
                return true;
            }
        }
    }
    return false;
}

bool operator==(const Mat &A, const Mat &B)
{
    if(A.getMatType()!=B.getMatType() || A.getWidth()!=B.getWidth() || A.getHeight()!=B.getHeight() || A.getChannel()!=B.getChannel())
    {
        return false;
    }

    if(A.isU8Mat())
    {
        for (size_t i = 0; i < A.getDataNum(); ++i)
        {
            if(A.getData().u8[i] != B.getData().u8[i])
            {
                return false;
            }
        }
    }

    if(A.isF32Mat())
    {
        for (size_t i = 0; i < A.getDataNum(); ++i)
        {
            if(fabsf(A.getData().f32[i] - B.getData().f32[i])>MSNH_F32_EPS)
            {
                return false;
            }
        }
    }

    if(A.isF64Mat())
    {
        for (size_t i = 0; i < A.getDataNum(); ++i)
        {
            if(std::fabs(A.getData().f64[i] - B.getData().f64[i])>MSNH_F64_EPS)
            {
                return false;
            }
        }
    }
    return true;
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
        throw Exception(1,"[Mat]: img empty! \n", __FILE__, __LINE__, __FUNCTION__);
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
            uint64_t dataLen = this->_height*this->_width*this->_channel;
            uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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

Mat Mat::toFloat32()
{
    if(this->isEmpty())
    {
        throw Exception(1,"[Mat]: img empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat mat;
    convertTo(mat, CvtDataType::CVT_DATA_TO_F32_DIRECTLY);
    return mat;
}

Mat Mat::toFloat64()
{
    if(this->isEmpty())
    {
        throw Exception(1,"[Mat]: img empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat mat;
    convertTo(mat, CvtDataType::CVT_DATA_TO_F64_DIRECTLY);
    return mat;
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

void Mat::setDataNull()
{
    this->_width    = 0;
    this->_height   = 0;
    this->_channel  = 0;
    this->_step     = 0;
    this->_matType  = MatType::MAT_RGB_U8;
    this->_data.u8  = nullptr;
}

uint8_t *Mat::getBytes() const
{
    return this->_data.u8;
}

float *Mat::getFloat32() const
{
    return this->_data.f32;
}

double *Mat::getFloat64() const
{
    return this->_data.f64;
}

double Mat::getVal2Double(const size_t &index) const
{
    if(index >= this->getDataNum())
    {
        throw Exception(1,"[Mat]: index out of Memory! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(this->isU8Mat())
    {
        return this->getBytes()[index];
    }
    else if(this->isF32Mat())
    {
        return this->getFloat32()[index];
    }
    else  

    {
        return this->getFloat64()[index];
    }
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

Vec2I32 Mat::getSize() const
{
    return Vec2I32(this->_width, this->_height);
}

size_t Mat::getDataNum() const
{
    return this->_width*this->_height*this->_channel;
}

size_t Mat::getByteNum() const
{
    return this->_width*this->_height*this->_step;
}

uint8_t Mat::getPerDataByteNum() const
{
    return static_cast<uint8_t>(this->_step/this->_channel);
}

Mat Mat::eye(const int &num, const MatType &matType)
{
    if(matType != MAT_GRAY_U8 && matType != MAT_GRAY_F32 && matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[Mat]: Only one channel is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmp(num,num, matType);

    if(matType == MAT_GRAY_U8)
    {
#ifdef USE_OMP
        uint64_t dataLen   = tmp.getHeight();
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().u8[i*tmp.getWidth()+i] = 1;
        }
    }
    else if(matType == MAT_GRAY_F32)
    {
#ifdef USE_OMP
        uint64_t dataLen   = tmp.getHeight();
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().f32[i*tmp.getWidth()+i] = 1.f;
        }
    }
    else if(matType == MAT_GRAY_F64)
    {
#ifdef USE_OMP
        uint64_t dataLen   = tmp.getHeight();
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        throw Exception(1,"[Mat]: Only one channel is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmp(width,height, matType);

    if(matType == MAT_GRAY_U8)
    {
#ifdef USE_OMP
        uint64_t dataLen   = tmp.getHeight()*tmp.getWidth();;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = tmp.getHeight()*tmp.getWidth();;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = tmp.getHeight()*tmp.getWidth();
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        throw Exception(1,"[Mat]: Only one channel is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmp(num,num, matType);

    if(matType == MAT_GRAY_U8)
    {
#ifdef USE_OMP
        uint64_t dataLen   = tmp.getHeight();
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().u8[i*tmp.getWidth()+i] = static_cast<uint8_t>(val) ;
        }
    }
    else if(matType == MAT_GRAY_F32)
    {
#ifdef USE_OMP
        uint64_t dataLen   = tmp.getHeight();
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < tmp.getHeight(); ++i)
        {
            tmp.getData().f32[i*tmp.getWidth()+i] = val;
        }
    }
    else if(matType == MAT_GRAY_F64)
    {
#ifdef USE_OMP
        uint64_t dataLen   = tmp.getHeight();
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        throw Exception(1,"[Mat]: Only one channel is supported! \n", __FILE__, __LINE__, __FUNCTION__);
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
        throw Exception(1,"[Mat]: Only one channel is supported! \n", __FILE__, __LINE__, __FUNCTION__);
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

Mat Mat::transpose() const
{
    if(this->isEmpty())
    {
        throw Exception(1,"[Mat]: Mat empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(!this->isOneChannel())
    {
        throw Exception(1,"[Mat]: Only one channel is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat(this->_height,this->_width,this->_matType);

    if(isU8Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = this->_height*this->_width;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = this->_height*this->_width;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = this->_height*this->_width;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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

double Mat::det() const
{
    if(this->_matType != MAT_GRAY_F32 && this->_matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[Mat]: Only one channel F32 and F64 Mat is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(this->_height!=this->_width)
    {
        throw Exception(1,"[Mat]: Height must equal width! \n", __FILE__, __LINE__, __FUNCTION__);
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

                    if(abs(alpha)<MSNH_F32_EPS)
                    {
                        continue;
                    }

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

                    if(abs(alpha)<MSNH_F64_EPS)
                    {
                        continue;
                    }

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

double Mat::trace() const
{

    if(this->_matType != MAT_GRAY_F32 && this->_matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[Mat]: Only one channel F32/F64 Mat is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    int tmp = std::min(_width,_height);

    double tr = 0;

    if(this->_matType == MAT_GRAY_F32)
    {
        for (int i = 0; i < tmp; ++i)
        {
            tr += getFloat32()[i*_width+i];
        }
    }
    else
    {
        for (int i = 0; i < tmp; ++i)
        {
            tr += getFloat64()[i*_width+i];
        }
    }
    return tr;
}

std::vector<Mat> Mat::LUDecomp(bool outLU) const
{
    if(this->_matType != MAT_GRAY_F32 && this->_matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[Mat]: Only one channel F32/F64 Mat is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(this->_height!=this->_width)
    {
        throw Exception(1,"[Mat]: Height must equal width! \n", __FILE__, __LINE__, __FUNCTION__);
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
                throw Exception(1,"[Mat]: det=0, no invert mat! \n", __FILE__, __LINE__, __FUNCTION__);

            if( k != i )
            {
                for(int j = i; j < m; j++ )
                {
                    std::swap(A.getData().f32[i*m + j], A.getData().f32[k*m + j]);
                }

                if(!outLU)
                {
                    for(int j = 0; j < m; j++ )
                    {
                        std::swap(B.getData().f32[i*m + j], B.getData().f32[k*m + j]);
                    }
                }

            }

            float d = -1/A.getData().f32[i*m + i];

            for(int j = i+1; j < m; j++ )
            {
                float alpha = A.getData().f32[j*m + i]*d;

                if(abs(alpha)<MSNH_F32_EPS)
                {
                    continue;
                }

                if(!outLU)
                {

                    for( k = i+1; k < m; k++ )
                    {
                        A.getData().f32[j*m + k] += alpha*A.getData().f32[i*m + k];
                    }

                    for( k = 0; k < m; k++ )
                    {
                        B.getData().f32[j*m + k] += alpha*B.getData().f32[i*m + k];
                    }
                }
                else
                {

                    for( k = i; k < m; k++ )
                    {
                        if(k < j)
                            B.getData().f32[j*m + k] = -d*A.getData().f32[j*m + k];
                        A.getData().f32[j*m + k] += alpha*A.getData().f32[i*m + k];
                    }

                }

            }
            if(!outLU)
                A.getData().f32[i*m + i] = -d;  

        }

        if(!outLU)
        {
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
                throw Exception(1,"[Mat]: det=0, no invert mat! \n", __FILE__, __LINE__, __FUNCTION__);

            if( k != i )
            {
                for(int j = i; j < m; j++ ) 

                {
                    std::swap(A.getData().f64[i*m + j], A.getData().f64[k*m + j]);
                }

                if(!outLU)

                {
                    for(int j = 0; j < m; j++ )

                    {
                        std::swap(B.getData().f64[i*m + j], B.getData().f64[k*m + j]);
                    }
                }
            }

            double d = -1/A.getData().f64[i*m + i];

            for(int j = i+1; j < m; j++ )
            {
                double alpha = A.getData().f64[j*m + i]*d;

                if(abs(alpha)<MSNH_F64_EPS)
                {
                    continue;
                }

                if(!outLU)
                {

                    for( k = i+1; k < m; k++ )
                    {
                        A.getData().f64[j*m + k] += alpha*A.getData().f64[i*m + k];
                    }

                    for( k = 0; k < m; k++ )
                    {
                        B.getData().f64[j*m + k] += alpha*B.getData().f64[i*m + k];
                    }
                }
                else
                {

                    for( k = i; k < m; k++ )
                    {
                        if(k < j)
                            B.getData().f64[j*m + k] = -d*A.getData().f64[j*m + k];
                        A.getData().f64[j*m + k] += alpha*A.getData().f64[i*m + k];
                    }
                }

            }
            if(!outLU)
                A.getData().f64[i*m + i] = -d;  

        }

        if(!outLU)
        {
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
    }

    if(outLU)
    {
        std::vector<Mat> tmpMatVec{B,A};
        return tmpMatVec;
    }
    else
    {
        std::vector<Mat> tmpMatVec{B};
        return tmpMatVec;
    }
}

std::vector<Mat> Mat::choleskyDeComp(bool outChols) const
{
    if(this->_matType != MAT_GRAY_F32 && this->_matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[Mat]: Only one channel F32/F64 Mat is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(this->_height!=this->_width)
    {
        throw Exception(1,"[Mat]: Height must equal width! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat A   = *this;
    Mat L   = Mat::eye(A.getWidth(), A.getMatType());
    Mat eye = Mat::eye(A.getWidth(), A.getMatType());
    int m   = A.getWidth();

    if(this->_matType == MAT_GRAY_F32)
    {
        float s = 0;
        for(int i = 0; i < m; i++ )
        {

            for(int j = 0; j < i; j++ )
            {
                s = A.getData().f32[j*m + i];
                for(int k = 0; k < j; k++ )
                {
                    s -= A.getData().f32[i*m + k]*A.getData().f32[j*m + k];
                }
                A.getData().f32[i*m + j] = s/A.getData().f32[j*m + j];
                L.getData().f32[i*m + j] = s/A.getData().f32[j*m + j];
            }

            s = A.getData().f32[i*m + i];
            for(int k = 0; k < i; k++ )
            {
                float t =A.getData().f32[i*m + k];
                s -= t*t;
            }
            if( s < std::numeric_limits<float>::epsilon() )
                throw Exception(1,"[Mat]: Not a good matrix for cholesky! \n", __FILE__, __LINE__, __FUNCTION__);
            A.getData().f32[i*m + i] = std::sqrt(s);
            L.getData().f32[i*m + i] = std::sqrt(s);

            if(!outChols)
            {
                L.getData().f32[i*m + i] = 1/L.getData().f32[i*m + i];
            }
        }

        if(!outChols)
        {

            for(int i = 0; i < m; i++ )
            {
                for(int j = 0; j < m; j++ )
                {
                    float s = eye.getData().f32[i*m + j];
                    for( int k = 0; k < i; k++ )
                    {
                        s -= L.getData().f32[i*m + k]*eye.getData().f32[k*m + j];
                    }
                    eye.getData().f32[i*m + j] = s*L.getData().f32[i*m + i];
                }
            }

            for(int i = m-1; i >=0; i-- )
            {
                for(int j = 0; j < m; j++ )
                {
                    float s = eye.getData().f32[i*m + j];
                    for( int k = m-1; k > i; k-- )
                    {
                        s -= L.getData().f32[k*m + i]*eye.getData().f32[k*m + j];
                    }
                    eye.getData().f32[i*m + j] = s*L.getData().f32[i*m + i];
                }
            }
        }
    }
    else
    {
        double s = 0;
        for(int i = 0; i < m; i++ )
        {

            for(int j = 0; j < i; j++ )
            {
                s = A.getData().f64[j*m + i];
                for(int k = 0; k < j; k++ )
                {
                    s -= A.getData().f64[i*m + k]*A.getData().f64[j*m + k];
                }
                A.getData().f64[i*m + j] = s/A.getData().f64[j*m + j];
                L.getData().f64[i*m + j] = s/A.getData().f64[j*m + j];
            }

            s = A.getData().f64[i*m + i];
            for(int k = 0; k < i; k++ )
            {
                double t =A.getData().f64[i*m + k];
                s -= t*t;
            }
            if( s < std::numeric_limits<double>::epsilon() )
                throw Exception(1,"[Mat]: Not a good matrix for cholesky! \n", __FILE__, __LINE__, __FUNCTION__);
            A.getData().f64[i*m + i] = std::sqrt(s);
            L.getData().f64[i*m + i] = std::sqrt(s);

            if(!outChols)
            {
                L.getData().f64[i*m + i] = 1/L.getData().f64[i*m + i];
            }
        }

        if(!outChols)
        {

            for(int i = 0; i < m; i++ )
            {
                for(int j = 0; j < m; j++ )
                {
                    double s = eye.getData().f64[i*m + j];
                    for( int k = 0; k < i; k++ )
                    {
                        s -= L.getData().f64[i*m + k]*eye.getData().f64[k*m + j];
                    }
                    eye.getData().f64[i*m + j] = s*L.getData().f64[i*m + i];
                }
            }

            for(int i = m-1; i >=0; i-- )
            {
                for(int j = 0; j < m; j++ )
                {
                    double s = eye.getData().f64[i*m + j];
                    for( int k = m-1; k > i; k-- )
                    {
                        s -= L.getData().f64[k*m + i]*eye.getData().f64[k*m + j];
                    }
                    eye.getData().f64[i*m + j] = s*L.getData().f64[i*m + i];
                }
            }
        }
    }

    if(outChols)
    {
        std::vector<Mat> tmpMatVec{L,L.transpose()};
        return tmpMatVec;
    }
    else
    {

        std::vector<Mat> tmpMatVec{eye};
        return tmpMatVec;
    }
}

std::vector<Mat> Mat::eigen(bool sort, bool forceCheckSymmetric)
{
    if(this->getWidth()!=this->getHeight())
    {
        throw Exception(1,"[Mat]: mat must be square and should be a symmetric matrix! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(forceCheckSymmetric)
    {
        if(this->getMatType() == MAT_GRAY_F32)
        {
            for (int i = 0; i < this->getHeight(); ++i)
            {
                for (int j = i+1; j < this->getWidth(); ++j)
                {
                    if(this->getFloat32()[i*this->getWidth()+j]!=this->getFloat32()[j*this->getWidth()+i])
                    {
                        throw Exception(1,"[Mat]: not a symmetric matrix! \n", __FILE__, __LINE__, __FUNCTION__);
                    }
                }
            }
        }
        else if(this->getMatType() == MAT_GRAY_F64)
        {
            for (int i = 0; i < this->getHeight(); ++i)
            {
                for (int j = i+1; j < this->getWidth(); ++j)
                {
                    if(this->getFloat64()[i*this->getWidth()+j]!=this->getFloat64()[j*this->getWidth()+i])
                    {
                        throw Exception(1,"[Mat]: not a symmetric matrix! \n", __FILE__, __LINE__, __FUNCTION__);
                    }
                }
            }
        }
    }

    int n           = this->getWidth(); 

    MatType matType = this->getMatType();

    Mat eigenvalues(n,1,matType);
    Mat V = Mat::eye(n,matType);
    Mat A = (*this);
    std::vector<int> indR(n, 0);
    std::vector<int> indC(n, 0);
    int maxIters = n*n*30; 

    if(this->getMatType() == MAT_GRAY_F32)
    {
        float maxVal = 0;

        for (int i = 0; i < n; ++i)
        {
            eigenvalues.getFloat32()[i] = this->getFloat32()[i*n+i];
        }

        for (int k = 0; k < n; ++k)
        {
            int maxIdx   = 0;
            int i   = 0;

            if (k < n - 1)
            {
                for (maxIdx = k + 1, maxVal = std::abs(A.getFloat32()[n*k + maxIdx]), i = k + 2; i < n; i++)
                {
                    float val = std::abs(A.getFloat32()[n*k + i]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        maxIdx = i;
                    }
                }
                indR[k] = maxIdx;
            }

            if (k > 0)
            {
                for (maxIdx = 0, maxVal = std::abs(A.getFloat32()[k]), i = 1; i < k; i++)
                {
                    float val = std::abs(A.getFloat32()[n*i + k]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        maxIdx = i;
                    }
                }

                indC[k] = maxIdx;
            }
        }

        if (n > 1)
        {
            for (int iters = 0; iters < maxIters; iters++)
            {
                int k   =   0;
                int i   =   0;
                int m   =   0;

                for (k = 0, maxVal = std::abs(A.getFloat32()[indR[0]]), i = 1; i < n - 1; i++)
                {
                    float val = std::abs(A.getFloat32()[n*i + indR[i]]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        k      = i;
                    }
                }

                int l = indR[k];
                for (i = 1; i < n; i++)
                {
                    float val = std::abs(A.getFloat32()[n*indC[i] + i]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        k = indC[i];
                        l = i;
                    }
                }

                float p = A.getFloat32()[n*k + l];

                if (std::abs(p) <= MSNH_F32_EPS)
                    break;
                float y = ((eigenvalues.getFloat32()[l] - eigenvalues.getFloat32()[k])*0.5f);
                float t = std::abs(y) + hypot(p, y);
                float s = hypot(p, t);
                float c = t / s;

                s = p / s;
                t = (p / t)*p;

                if (y < 0)
                {
                    s = -s;
                    t = -t;
                }

                A.getFloat32()[n*k + l] = 0;

                eigenvalues.getFloat32()[k] -= t;
                eigenvalues.getFloat32()[l] += t;

                float a0    =   0;
                float b0    =   0;

#undef rotate
#define rotate(v0, v1) (a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c)

                for (i = 0; i < k; i++)
                {
                    rotate(A.getFloat32()[n*i + k], A.getFloat32()[n*i + l]);
                }

                for (i = k + 1; i < l; i++)
                {
                    rotate(A.getFloat32()[n*k + i], A.getFloat32()[n*i + l]);
                }

                for (i = l + 1; i < n; i++)
                {
                    rotate(A.getFloat32()[n*k + i], A.getFloat32()[n*l + i]);
                }

                for (i = 0; i < n; i++)
                {
                    rotate(V.getFloat32()[n*k + i], V.getFloat32()[n*l + i]);
                }

#undef rotate

                for (int j = 0; j < 2; j++)
                {
                    int idx = j == 0 ? k : l;
                    if (idx < n - 1)
                    {
                        for (m = idx + 1, maxVal = std::abs(A.getFloat32()[n*idx + m]), i = idx + 2; i < n; i++)
                        {
                            float val = std::abs(A.getFloat32()[n*idx + i]);

                            if (maxVal < val)
                            {
                                maxVal = val;
                                m = i;
                            }
                        }
                        indR[idx] = m;
                    }
                    if (idx > 0)
                    {
                        for (m = 0, maxVal = std::abs(A.getFloat32()[idx]), i = 1; i < idx; i++)
                        {
                            float val = std::abs(A.getFloat32()[n*i + idx]);

                            if (maxVal < val)
                            {
                                maxVal = val;
                                m = i;
                            }
                        }
                        indC[idx] = m;
                    }
                }
            }
        }

        if (sort)
        {
            for (int k = 0; k < n - 1; k++)
            {
                int m = k;
                for (int i = k + 1; i < n; i++)
                {
                    if (std::abs(eigenvalues.getFloat32()[m]) < std::abs(eigenvalues.getFloat32()[i]))
                        m = i;
                }

                if (k != m)
                {
                    std::swap(eigenvalues.getFloat32()[m], eigenvalues.getFloat32()[k]);

                    for (int i = 0; i < n; i++)
                    {
                        std::swap(V.getFloat32()[n*m + i], V.getFloat32()[n*k + i]);
                    }
                }
            }
        }

        return std::vector<Mat>{eigenvalues,V.transpose()};
    }
    else if(this->getMatType() == MAT_GRAY_F64)
    {
        double maxVal = 0;

        for (int i = 0; i < n; ++i)
        {
            eigenvalues.getFloat64()[i] = this->getFloat64()[i*n+i];
        }

        for (int k = 0; k < n; ++k)
        {
            int maxIdx   = 0;
            int i   = 0;

            if (k < n - 1)
            {
                for (maxIdx = k + 1, maxVal = std::abs(A.getFloat64()[n*k + maxIdx]), i = k + 2; i < n; i++)
                {
                    double val = std::abs(A.getFloat64()[n*k + i]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        maxIdx = i;
                    }
                }
                indR[k] = maxIdx;
            }

            if (k > 0)
            {
                for (maxIdx = 0, maxVal = std::abs(A.getFloat64()[k]), i = 1; i < k; i++)
                {
                    double val = std::abs(A.getFloat64()[n*i + k]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        maxIdx = i;
                    }
                }

                indC[k] = maxIdx;
            }
        }

        if (n > 1)
        {
            for (int iters = 0; iters < maxIters; iters++)
            {
                int k   =   0;
                int i   =   0;
                int m   =   0;

                for (k = 0, maxVal = std::abs(A.getFloat64()[indR[0]]), i = 1; i < n - 1; i++)
                {
                    double val = std::abs(A.getFloat64()[n*i + indR[i]]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        k      = i;
                    }
                }

                int l = indR[k];
                for (i = 1; i < n; i++)
                {
                    double val = std::abs(A.getFloat64()[n*indC[i] + i]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        k = indC[i];
                        l = i;
                    }
                }

                double p = A.getFloat64()[n*k + l];

                if (std::abs(p) <= MSNH_F64_EPS)
                    break;
                double y = ((eigenvalues.getFloat64()[l] - eigenvalues.getFloat64()[k])*0.5);
                double t = std::abs(y) + hypot(p, y);
                double s = hypot(p, t);
                double c = t / s;

                s = p / s;
                t = (p / t)*p;

                if (y < 0)
                {
                    s = -s;
                    t = -t;
                }

                A.getFloat64()[n*k + l] = 0;

                eigenvalues.getFloat64()[k] -= t;
                eigenvalues.getFloat64()[l] += t;

                double a0    =   0;
                double b0    =   0;

#undef rotate
#define rotate(v0, v1) (a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c)

                for (i = 0; i < k; i++)
                {
                    rotate(A.getFloat64()[n*i + k], A.getFloat64()[n*i + l]);
                }

                for (i = k + 1; i < l; i++)
                {
                    rotate(A.getFloat64()[n*k + i], A.getFloat64()[n*i + l]);
                }

                for (i = l + 1; i < n; i++)
                {
                    rotate(A.getFloat64()[n*k + i], A.getFloat64()[n*l + i]);
                }

                for (i = 0; i < n; i++)
                {
                    rotate(V.getFloat64()[n*k + i], V.getFloat64()[n*l + i]);
                }

#undef rotate

                for (int j = 0; j < 2; j++)
                {
                    int idx = j == 0 ? k : l;
                    if (idx < n - 1)
                    {
                        for (m = idx + 1, maxVal = std::abs(A.getFloat64()[n*idx + m]), i = idx + 2; i < n; i++)
                        {
                            double val = std::abs(A.getFloat64()[n*idx + i]);

                            if (maxVal < val)
                            {
                                maxVal = val;
                                m = i;
                            }
                        }
                        indR[idx] = m;
                    }
                    if (idx > 0)
                    {
                        for (m = 0, maxVal = std::abs(A.getFloat64()[idx]), i = 1; i < idx; i++)
                        {
                            double val = std::abs(A.getFloat64()[n*i + idx]);

                            if (maxVal < val)
                            {
                                maxVal = val;
                                m = i;
                            }
                        }
                        indC[idx] = m;
                    }
                }
            }
        }

        if (sort)
        {
            for (int k = 0; k < n - 1; k++)
            {
                int m = k;
                for (int i = k + 1; i < n; i++)
                {
                    if (std::abs(eigenvalues.getFloat64()[m]) < std::abs(eigenvalues.getFloat64()[i]))
                        m = i;
                }

                if (k != m)
                {
                    std::swap(eigenvalues.getFloat64()[m], eigenvalues.getFloat64()[k]);

                    for (int i = 0; i < n; i++)
                    {
                        std::swap(V.getFloat64()[n*m + i], V.getFloat64()[n*k + i]);
                    }
                }
            }
        }

        return std::vector<Mat>{eigenvalues,V.transpose()};
    }

}

std::vector<Mat> Mat::svd()
{

    if(this->getMatType() != MAT_GRAY_F32 && this->getMatType()!=MAT_GRAY_F64)
    {
        throw Exception(1,"[Mat]: Only one channel F32/F64 Mat is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    int n = this->getWidth(); 

    int m = this->getHeight();

    bool at = false;

    if(m<n)
    {
        at = true;
        std::swap(m, n);
    }

    Mat U(m,m, this->getMatType());
    Mat D(1,n,this->getMatType());
    Mat Vt(n,n,this->getMatType());

    Mat AMat;

    if(!at)
    {
        AMat = this->transpose();
    }
    else
    {
        AMat = (*this);
    }

    Mat AMatNew; 

    if(m == n)
    {
        AMatNew = AMat;
    }
    else
    {
        /*     x x x   ->  x x x
         *     x x x       x x x
         *                 0 0 0
         * */

        AMatNew = Mat(m,m,AMat.getMatType()); 

        memcpy(AMatNew.getBytes(), AMat.getBytes(), AMat.getByteNum());
    }

    if(this->isF32Mat())
    {
        jacobiSVD<float>(AMatNew, D, Vt);
    }
    else
    {
        jacobiSVD<double>(AMatNew, D, Vt);
    }

    if(!at)
    {
        U   = AMatNew.transpose();
    }
    else
    {
        U   = Vt.transpose();
        Vt  = AMatNew;
    }
    return std::vector<Mat>{U,D,Vt};
}

Mat Mat::pseudoInvert()
{

    if(this->_matType != MAT_GRAY_F32 && this->_matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[Mat]: Only one channel F32/F64 Mat is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    int m   = this->_height;
    int n   = this->_width;

    auto UDVT = this->svd();

    Mat V   = UDVT[2].transpose();
    Mat UT  = UDVT[0].transpose();

    if(m < n)
    {
        std::swap(m,n);
    }

    Mat Drecip(m,n,this->_matType);

    if(this->_matType == MAT_GRAY_F32)
    {
        for (int i = 0; i < n; ++i)
        {
            if(UDVT[1].getFloat32()[i]>MSNH_F32_EPS)
                Drecip.getFloat32()[i*m+i] = 1.0f/UDVT[1].getFloat32()[i];
        }
    }
    else
    {
        for (int i = 0; i < n; ++i)
        {
            if(UDVT[1].getFloat64()[i]>MSNH_F64_EPS)
                Drecip.getFloat64()[i*m+i] = 1.0/UDVT[1].getFloat64()[i];
        }
    }

    if(this->_height < this->_width)
    {
        Drecip = Drecip.transpose();
    }

    return V*Drecip*UT;

}

Mat Mat::invert(const DecompType &decompType) const
{
    if(this->_matType != MAT_GRAY_F32 && this->_matType != MAT_GRAY_F64)
    {
        throw Exception(1,"[Mat]: Only one channel F32/F64 Mat is supported! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(this->_height!=this->_width)
    {
        throw Exception(1,"[Mat]: Height must equal width! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    switch (decompType)
    {
    case DECOMP_LU:
        return LUDecomp(false)[0];
    case DECOMP_CHOLESKY:
        return choleskyDeComp(false)[0];
    default:
        return LUDecomp(false)[0];
    }

}

void Mat::print()
{
    std::cout<<"{  width: "<<this->_width<<" , height: "<<this->_height<<" , channels: "<<this->_channel<<" , type: "<<getMatTypeStr()<<std::endl;

    if(isF32Mat())
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            std::cout<<"    ["<<std::endl;
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<19|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==0)
                        {
                            std::cout<<"        ";
                        }

                        if(j==19)
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                        }
                        else if(j<19 || j==(this->_width-1) )
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<std::setiosflags(std::ios::fixed)<<this->_data.f32[c*this->_width*this->_height + i*this->_width + j]<<" ";
                        }
                    }
                    std::cout<<";"<<std::endl;
                }
                else if(i == 20)
                {
                    std::cout<<"        "<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                    std::cout<<";"<<std::endl;
                }
            }
            std::cout<<"    ],"<<std::endl;
        }
    }
    else if(isF64Mat())
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            std::cout<<"    ["<<std::endl;
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<9|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==0)
                        {
                            std::cout<<"        ";
                        }

                        if(j==9)
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                        }
                        else if(j<9 || j==(this->_width-1) )
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(12)<<std::setiosflags(std::ios::fixed)<<this->_data.f64[c*this->_width*this->_height + i*this->_width + j]<<" ";
                        }
                    }
                    std::cout<<";"<<std::endl;
                }
                else if(i == 10)
                {
                    std::cout<<"        "<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                    std::cout<<";"<<std::endl;
                }

            }
            std::cout<<"    ],"<<std::endl;
        }
    }
    else
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            std::cout<<"    ["<<std::endl;
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<19|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==0)
                        {
                            std::cout<<"        ";
                        }

                        if(j==19)
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(6)<<" ... ";
                        }
                        else if(j<19 || j==(this->_width-1) )
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(6)<<static_cast<int>(this->_data.u8[c*this->_width*this->_height + i*this->_width + j])<<std::setw(1)<<" ";
                        }
                    }
                    std::cout<<";"<<std::endl;
                }
                else if(i == 20)
                {
                    std::cout<<"        "<<std::setiosflags(std::ios::left)<<std::setw(6)<<" ... ";
                    std::cout<<";"<<std::endl;
                }
            }
            std::cout<<"    ],"<<std::endl;
        }
    }
    std::cout<<"}"<<std::endl<<std::endl;
}

string Mat::toString()
{
    std::stringstream buf;
    buf<<"{  width: "<<this->_width<<" , height: "<<this->_height<<" , channels: "<<this->_channel<<" , type: "<<getMatTypeStr()<<std::endl;

    if(isF32Mat())
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            buf<<"    ["<<std::endl;
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<19|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==0)
                        {
                            buf<<"        ";
                        }

                        if(j==19)
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                        }
                        else if(j<19 || j==(this->_width-1) )
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<std::setiosflags(std::ios::fixed)<<this->_data.f32[c*this->_width*this->_height + i*this->_width + j]<<" ";
                        }
                    }
                    buf<<";"<<std::endl;
                }
                else if(i == 20)
                {
                    buf<<"      "<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                    buf<<";"<<std::endl;
                }

            }
            buf<<"    ],"<<std::endl;
        }
    }
    else if(isF64Mat())
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            buf<<"    ["<<std::endl;
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<9|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==0)
                        {
                            buf<<"        ";
                        }

                        if(j==9)
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                        }
                        else if(j<9 || j==(this->_width-1) )
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(12)<<std::setiosflags(std::ios::fixed)<<this->_data.f64[c*this->_width*this->_height + i*this->_width + j]<<" ";
                        }
                    }
                    buf<<";"<<std::endl;
                }
                else if(i == 10)
                {
                    buf<<"      "<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                    buf<<";"<<std::endl;
                }

            }
            buf<<"    ],"<<std::endl;
        }
    }
    else
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            buf<<"    ["<<std::endl;
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<19|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==0)
                        {
                            buf<<"        ";
                        }

                        if(j==19)
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(6)<<" ... ";
                        }
                        else if(j<19 || j==(this->_width-1) )
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(6)<<static_cast<int>(this->_data.u8[c*this->_width*this->_height + i*this->_width + j])<<std::setw(1)<<" ";
                        }
                    }
                    buf<<";"<<std::endl;
                }
                else if(i == 20)
                {
                    buf<<"      "<<std::setiosflags(std::ios::left)<<std::setw(6)<<" ... ";
                    buf<<";"<<std::endl;
                }
            }
            buf<<"    ],"<<std::endl;
        }
    }
    buf<<"}"<<std::endl<<std::endl;
    return buf.str();
}

string Mat::toHtmlString()
{
    std::stringstream buf;
    buf<<"{  width: "<<this->_width<<" , height: "<<this->_height<<" , channels: "<<this->_channel<<" , type: "<<getMatTypeStr()<<"<br/>";

    if(isF32Mat())
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            buf<<"    ["<<"<br/>";
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<19|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==0)
                        {
                            buf<<"        ";
                        }

                        if(j==19)
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                        }
                        else if(j<19 || j==(this->_width-1) )
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<std::setiosflags(std::ios::fixed)<<this->_data.f32[c*this->_width*this->_height + i*this->_width + j]<<" ";
                        }
                    }
                    buf<<";"<<"<br/>";
                }
                else if(i == 20)
                {
                    buf<<"      "<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                    buf<<";"<<"<br/>";
                }

            }
            buf<<"    ],"<<"<br/>";
        }
    }
    else if(isF64Mat())
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            buf<<"    ["<<"<br/>";
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<9|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==0)
                        {
                            buf<<"        ";
                        }

                        if(j==9)
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                        }
                        else if(j<9 || j==(this->_width-1) )
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(12)<<std::setiosflags(std::ios::fixed)<<this->_data.f64[c*this->_width*this->_height + i*this->_width + j]<<" ";
                        }
                    }
                    buf<<";"<<"<br/>";
                }
                else if(i == 10)
                {
                    buf<<"      "<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                    buf<<";"<<"<br/>";
                }

            }
            buf<<"    ],"<<"<br/>";
        }
    }
    else
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            buf<<"    ["<<"<br/>";
            for (int i = 0; i < this->_height; ++i)
            {
                if(i<19|| (i==this->_height-1) )
                {
                    for (int j = 0; j < this->_width; ++j)
                    {
                        if(j==0)
                        {
                            buf<<"        ";
                        }

                        if(j==19)
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(6)<<" ... ";
                        }
                        else if(j<19 || j==(this->_width-1) )
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(6)<<static_cast<int>(this->_data.u8[c*this->_width*this->_height + i*this->_width + j])<<std::setw(1)<<" ";
                        }
                    }
                    buf<<";"<<"<br/>";
                }
                else if(i == 20)
                {
                    buf<<"      "<<std::setiosflags(std::ios::left)<<std::setw(6)<<" ... ";
                    buf<<";"<<"<br/>";
                }
            }
            buf<<"    ],"<<"<br/>";
        }
    }
    buf<<"}"<<"<br/>"<<"<br/>";
    return buf.str();
}

string Mat::getMatTypeStr()
{
    switch (_matType)
    {
    case MAT_GRAY_U8:
        return "MAT_GRAY_U8";
        break;
    case MAT_GRAY_F32:
        return "MAT_GRAY_F32";
        break;
    case MAT_GRAY_F64:
        return "MAT_GRAY_F64";
        break;
    case MAT_RGB_U8:
        return "MAT_RGB_U8";
        break;
    case MAT_RGB_F32:
        return "MAT_RGB_F32";
        break;
    case MAT_RGB_F64:
        return "MAT_RGB_F64";
        break;
    case MAT_RGBA_U8:
        return "MAT_RGBA_U8";
        break;
    case MAT_RGBA_F32:
        return "MAT_RGBA_F32";
        break;
    case MAT_RGBA_F64:
        return "MAT_RGBA_F64";
        break;
    default:
        return "Unknown";
        break;
    }
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

bool Mat::isVector() const
{
    return this->_channel==1 && (this->_width>1 && this->_height==1) || (this->_width==1 && this->_height>1);
}

bool Mat::isNum() const
{
    return (this->_channel==1 && this->_width==1 && this->_height==1);
}

bool Mat::isMatrix() const
{
    return (this->_width>1)&&(this->_height>1);
}

bool Mat::isMatrix3x3() const
{
    return (this->_width==3)&&(this->_height==3)&&(this->_channel==1);
}

bool Mat::isMatrix4x4() const
{
    return (this->_width==4)&&(this->_height==4)&&(this->_channel==1);
}

bool Mat::isRotMat() const
{
    if(this->_width!=3 || this->_height!=3 || this->_channel!=1 ||(this->_matType!=MAT_GRAY_F32 && this->_matType!=MAT_GRAY_F64))
    {
        return false;
    }

    if(this->_matType == MAT_GRAY_F64)
    {

        return Mat::eye(3,MatType::MAT_GRAY_F64)==(transpose()*(*this)) && (det()-1)<MSNH_F64_EPS;
    }
    else
    {

        return Mat::eye(3,MatType::MAT_GRAY_F32)==(transpose()*(*this)) && (det()-1)<MSNH_F32_EPS;
    }
}

bool Mat::isHomTransMatrix() const
{

    if(this->_width!=4 || this->_height!=4 || this->_channel!=1 ||(this->_matType!=MAT_GRAY_F32 && this->_matType!=MAT_GRAY_F64))
    {
        return false;
    }

    if(this->_matType == MAT_GRAY_F64)
    {
        if(std::abs(this->getFloat64()[12])>MSNH_F64_EPS || std::abs(this->getFloat64()[13])>MSNH_F64_EPS ||
                std::abs(this->getFloat64()[14])>MSNH_F64_EPS || std::abs(this->getFloat64()[15]-1)>MSNH_F64_EPS)
        {
            return false;
        }

        Mat R(3,3,MAT_GRAY_F64);
        R.getFloat64()[0] = this->getFloat64()[0];
        R.getFloat64()[1] = this->getFloat64()[1];
        R.getFloat64()[2] = this->getFloat64()[2];

        R.getFloat64()[3] = this->getFloat64()[4];
        R.getFloat64()[4] = this->getFloat64()[5];
        R.getFloat64()[5] = this->getFloat64()[6];

        R.getFloat64()[6] = this->getFloat64()[8];
        R.getFloat64()[7] = this->getFloat64()[9];
        R.getFloat64()[8] = this->getFloat64()[10];

        if(!R.isRotMat())
        {
            return false;
        }
        else
        {
            return true;
        }

    }
    else
    {
        if(fabsf(this->getFloat32()[12])>MSNH_F32_EPS || fabsf(this->getFloat32()[13])>MSNH_F32_EPS ||
                fabsf(this->getFloat32()[14])>MSNH_F32_EPS || fabsf(this->getFloat32()[15]-1)>MSNH_F32_EPS)
        {
            return false;
        }

        Mat R(3,3,MAT_GRAY_F32);
        R.getFloat32()[0] = this->getFloat32()[0];
        R.getFloat32()[1] = this->getFloat32()[1];
        R.getFloat32()[2] = this->getFloat32()[2];

        R.getFloat32()[3] = this->getFloat32()[4];
        R.getFloat32()[4] = this->getFloat32()[5];
        R.getFloat32()[5] = this->getFloat32()[6];

        R.getFloat32()[6] = this->getFloat32()[8];
        R.getFloat32()[7] = this->getFloat32()[9];
        R.getFloat32()[8] = this->getFloat32()[10];

        if(!R.isRotMat())
        {
            return false;
        }
        else
        {
            return true;
        }
    }
}

Mat Mat::add(const Mat &A, const Mat &B)
{
    return A+B;
}

Mat Mat::sub(const Mat &A, const Mat &B)
{
    return A-B;
}

Mat Mat::mul(const Mat &A, const Mat &B)
{
    return A*B;
}

Mat Mat::div(const Mat &A, const Mat &B)
{
    return A*B.invert();
}

Mat Mat::eleWiseDiv(const Mat &A, const Mat &B)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat B is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(A._matType != B._matType || A._channel != B._channel || A._step != B._step ||
            A._width != B._width || A._height != B._height)
    {
        throw Exception(1,"[Mat]: mat properties not equal! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    size_t dataN = A.getDataNum();

    if(tmpMat.isU8Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i] / B._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i] / B._data.f64[i];
        }
    }
    return tmpMat;
}

Mat Mat::eleWiseMul(const Mat &A, const Mat &B)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat B is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(A._matType != B._matType || A._channel != B._channel || A._step != B._step ||
            A._width != B._width || A._height != B._height)
    {
        throw Exception(1,"[Mat]: mat properties not equal! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    size_t dataN = A.getDataNum();

    if(tmpMat.isU8Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i] * B._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i] * B._data.f64[i];
        }
    }
    return tmpMat;
}

double Mat::dotProduct(const Mat &A, const Mat &B)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat B is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(A._matType != B._matType || A._channel != B._channel || A._step != B._step ||
            A._width != B._width || A._height != B._height)
    {
        throw Exception(1,"[Mat]: properties not equal! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    size_t dataN = A.getDataNum();

    if(A.isU8Mat())
    {
        uint8_t finalVal = 0;
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum) reduction(+:finalVal)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int mul = A._data.u8[i] * B._data.u8[i];

            finalVal += mul;
        }
        return finalVal;
    }
    else if(A.isF32Mat())
    {
        float finalVal = 0;
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum) reduction(+:finalVal)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            float mul= A._data.f32[i] * B._data.f32[i];
            finalVal += mul;
        }
        return finalVal;
    }
    else if(A.isF64Mat())
    {
        double finalVal = 0;
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum) reduction(+:finalVal)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            double mul = A._data.f64[i] * B._data.f64[i];
            finalVal += mul;
        }
        return finalVal;
    }
}

bool Mat::isNull() const
{
    if(this == nullptr)
    {
        return true;
    }

    for (size_t i = 0; i < this->getByteNum(); ++i)
    {
        if(this->getBytes()[i] > 0)
        {
            return false;
        }
    }
    return true;
}

bool Mat::isFuzzyNull() const
{
    if(this == nullptr)
    {
        return true;
    }

    if(this->isU8Mat())
    {
        for (size_t i = 0; i < this->getDataNum(); ++i)
        {
            if(this->getBytes()[i] > 0)
            {
                return false;
            }
        }
        return true;
    }
    else if(this->isF32Mat())
    {
        for (size_t i = 0; i < this->getDataNum(); ++i)
        {
            if(fabsf(this->getFloat32()[i]) > MSNH_F32_EPS)
            {
                return false;
            }
        }
        return true;
    }
    else
    {
        for (size_t i = 0; i < this->getDataNum(); ++i)
        {
            if(std::fabs(this->getFloat64()[i]) > MSNH_F64_EPS)
            {
                return false;
            }
        }
        return true;
    }
}

bool Mat::isNan() const
{
    if(this->isU8Mat())
    {
        for (size_t i = 0; i < this->getDataNum(); ++i)
        {
            if(std::isnan(static_cast<double>(this->getBytes()[i])))
            {
                return true;
            }
        }
        return false;
    }
    else if(this->isF32Mat())
    {
        for (size_t i = 0; i < this->getDataNum(); ++i)
        {
            if(std::isnan(static_cast<double>(this->getFloat32()[i])))
            {
                return true;
            }
        }
        return false;
    }
    else
    {
        for (size_t i = 0; i < this->getDataNum(); ++i)
        {
            if(std::isnan(this->getFloat64()[i]))
            {
                return true;
            }
        }
        return false;
    }
}

Mat operator +(const Mat &A, const Mat &B)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat B is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(A._matType != B._matType || A._channel != B._channel || A._step != B._step ||
            A._width != B._width || A._height != B._height)
    {
        throw Exception(1,"[Mat]: mat properties not equal! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    size_t dataN = A.getDataNum();

    if(tmpMat.isU8Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]+B._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    size_t dataN = A.getDataNum();

    if(tmpMat.isU8Mat())
    {
        int addVal = static_cast<int>(a);
        addVal = addVal>255?255:addVal;

#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]+ static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(tmpMat.isEmpty())
    {
        throw Exception(1,"[Mat]: mat empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    size_t dataN = A.getDataNum();

    if(tmpMat.isU8Mat())
    {
        int addVal = static_cast<int>(a);
        addVal = addVal>255?255:addVal;

#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]+ static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f64[i] = A._data.f64[i]+ a;
        }
    }
    return tmpMat;
}

Mat operator-(const Mat &A)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    return 0-A;
}

Mat operator -(const Mat &A, const Mat &B)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat B is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(A._matType != B._matType || A._channel != B._channel || A._step != B._step ||
            A._width != B._width || A._height != B._height)
    {
        throw Exception(1,"[Mat]: mat properties not equal! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    size_t dataN = A.getDataNum();

    if(tmpMat.isU8Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]-B._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(tmpMat.isEmpty())
    {
        throw Exception(1,"[Mat]: mat empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    size_t dataN = A.getDataNum();

    if(tmpMat.isU8Mat())
    {
        int subVal = static_cast<int>(a);
        subVal = subVal>255?255:subVal;
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = static_cast<float>(a) - A._data.f32[i];
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    if(tmpMat.isEmpty())
    {
        throw Exception(1,"[Mat]: mat empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    size_t dataN = A.getDataNum();

    if(tmpMat.isU8Mat())
    {
        int subVal = static_cast<int>(a);
        subVal = subVal>255?255:subVal;
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i] - static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat B is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(A._matType != B._matType)
    {
        throw Exception(1,"[Mat]: Mat type must be equal! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(A.isNum() && !B.isNum())  

    {
        return A.getVal2Double(0)*B;
    }
    else if(!A.isNum() && B.isNum())

    {
        return A*B.getVal2Double(0);
    }
    else if(A.isNum() && B.isNum())

    {
        if(A.isU8Mat())
        {
            uint8_t val = A.getBytes()[0]*B.getBytes()[0];
            return Mat(1,1,MAT_GRAY_U8,&val);
        }
        else if(A.isF32Mat())
        {
            float val = A.getFloat32()[0]*B.getFloat32()[0];
            return Mat(1,1,MAT_GRAY_F32,&val);
        }
        else if(A.isF64Mat())
        {
            double val = A.getFloat64()[0]*B.getFloat64()[0];
            return Mat(1,1,MAT_GRAY_F64,&val);
        }
    }
    else  

    {
        if(A.isVector()&&B.isVector())

        {

            if(A._matType != B._matType || A._channel != B._channel || A._step != B._step ||
                    A._width != B._width || A._height != B._height)
            {
                throw Exception(1,"[Mat]: properties not equal! \n", __FILE__, __LINE__, __FUNCTION__);
            }

            size_t dataN = A.getDataNum();

            if(A.isU8Mat())
            {
                uint8_t finalVal = 0;
#ifdef USE_OMP
                uint64_t dataLen   = dataN;
                uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)  reduction(+:finalVal)
#endif
                for (int i = 0; i < dataN; ++i)
                {
                    int mul = A._data.u8[i] * B._data.u8[i];

                    finalVal += mul;
                }
                return Mat(1,1,MAT_GRAY_U8, &finalVal);
            }
            else if(A.isF32Mat())
            {
                float finalVal = 0;
#ifdef USE_OMP
                uint64_t dataLen   = dataN;
                uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)  reduction(+:finalVal)
#endif
                for (int i = 0; i < dataN; ++i)
                {
                    float mul= A._data.f32[i] * B._data.f32[i];
                    finalVal += mul;
                }
                return Mat(1,1,MAT_GRAY_F32, &finalVal);
            }
            else if(A.isF64Mat())
            {
                double finalVal = 0;
#ifdef USE_OMP
                uint64_t dataLen   = dataN;
                uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)  reduction(+:finalVal)
#endif
                for (int i = 0; i < dataN; ++i)
                {
                    double mul = A._data.f64[i] * B._data.f64[i];
                    finalVal += mul;
                }
                return Mat(1,1,MAT_GRAY_F64, &finalVal);
            }

        }
        else  

        {
            if(A.getMatType() != B.getMatType() || A.getChannel() != B.getChannel() || A.getStep() != B.getStep())
            {
                throw Exception(1,"[Mat]: mat properties not equal! \n", __FILE__, __LINE__, __FUNCTION__);
            }

            if(!A.isF32Mat() && !A.isF64Mat() && !A.isOneChannel())
            {
                throw Exception(1,"[Mat]: mat must be f32/f64 1 channel mat! \n", __FILE__, __LINE__, __FUNCTION__);
            }

            if(A.getWidth() != B.getHeight())
            {
                throw Exception(1,"[Mat]: Mat A'W != B'H ! \n", __FILE__, __LINE__, __FUNCTION__);
            }

            SimdInfo::checkSimd();

            if(A.isF32Mat())
            {

                Mat C(B.getWidth(),A.getHeight(),MatType::MAT_GRAY_F32);
#ifdef USE_X86
                Gemm::cpuGemm(0,0,A.getHeight(),B.getWidth(),A.getWidth(),1,A.getData().f32,A.getWidth(),B.getData().f32,B.getWidth(),1,C.getData().f32,C.getWidth(), SimdInfo::supportAVX2);
#else
                Gemm::cpuGemm(0,0,A.getHeight(),B.getWidth(),A.getWidth(),1,A.getData().f32,A.getWidth(),B.getData().f32,B.getWidth(),1,C.getData().f32,C.getWidth(), false);
#endif
                return C;
            }
            else
            {
                Mat C(B.getWidth(),A.getHeight(),MatType::MAT_GRAY_F64);
#ifdef USE_X86
                Gemm::cpuGemm(0,0,A.getHeight(),B.getWidth(),A.getWidth(),1,A.getData().f64,A.getWidth(),B.getData().f64,B.getWidth(),1,C.getData().f64,C.getWidth(), SimdInfo::supportAVX2);
#else
                Gemm::cpuGemm(0,0,A.getHeight(),B.getWidth(),A.getWidth(),1,A.getData().f64,A.getWidth(),B.getData().f64,B.getWidth(),1,C.getData().f64,C.getWidth(), false);
#endif
                return C;
            }
        }
    }

}

Mat operator *(const double &a, const Mat &A)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    size_t dataN = A.getDataNum();

    if(tmpMat.isU8Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int mul = static_cast<int>(A._data.u8[i]* a);

            mul = (mul>255)?255:mul;
            mul = (mul<0)?0:mul;

            tmpMat._data.u8[i] = static_cast<uint8_t>(mul);
        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]*static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    Mat tmpMat = A;

    size_t dataN = A.getDataNum();

    if(tmpMat.isU8Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            int mul = static_cast<int>(A._data.u8[i]* a);

            mul = (mul>255)?255:mul;
            mul = (mul<0)?0:mul;

            tmpMat._data.u8[i] = static_cast<uint8_t>(mul);

        }
    }
    else if(tmpMat.isF32Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < dataN; ++i)
        {
            tmpMat._data.f32[i] = A._data.f32[i]*static_cast<float>(a);
        }
    }
    else if(tmpMat.isF64Mat())
    {
#ifdef USE_OMP
        uint64_t dataLen   = dataN;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
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
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(B.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat B is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(A._matType != B._matType)
    {
        throw Exception(1,"[Mat]: Mat type must be equal! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(A.isNum() && !B.isNum())  

    {
        return A.getVal2Double(0)/B;
    }
    else if(!A.isNum() && B.isNum())

    {
        return A*(1.0/B.getVal2Double(0));
    }
    else if(A.isNum() && B.isNum())

    {
        if(A.isU8Mat())
        {
            uint8_t val = A.getBytes()[0]/B.getBytes()[0];
            return Mat(1,1,MAT_GRAY_U8,&val);
        }
        else if(A.isF32Mat())
        {
            float val = A.getFloat32()[0]/B.getFloat32()[0];
            return Mat(1,1,MAT_GRAY_F32,&val);
        }
        else/* if(A.isF64Mat())*/
        {
            double val = A.getFloat64()[0]/B.getFloat64()[0];
            return Mat(1,1,MAT_GRAY_F64,&val);
        }
    }
    else
    {
        if(A.isVector()&&B.isVector())

        {
            throw Exception(1,"[Mat]: Vectors can't div! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        else
        {

            return A*B.invert(); /*B的逆现在只支持LU分解的逆*/
        }
    }
}

Mat operator /(const double &a, const Mat &A)
{
    if(A.isEmpty())
    {
        throw Exception(1,"[Mat]: Mat A is empty! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    if(A.isNum())
    {
        if(A.isU8Mat())
        {
            uint8_t val = static_cast<uint8_t>(a/A.getBytes()[0]);
            return Mat(1,1,MAT_GRAY_U8,&val);
        }
        else if(A.isF32Mat())
        {
            float val = static_cast<float>(a/A.getFloat32()[0]);
            return Mat(1,1,MAT_GRAY_F32,&val);
        }
        else/* if(A.isF64Mat())*/
        {
            double val = a/A.getFloat64()[0];
            return Mat(1,1,MAT_GRAY_F64,&val);
        }
    }
    else if(A.isVector())
    {
        throw Exception(1,"[Mat]: Vectors can't div! \n", __FILE__, __LINE__, __FUNCTION__);
    }
    else
    {

        return a*A.invert(); /*A的逆现在只支持LU分解的逆*/
    }
}

Mat operator /(const Mat &A, const double &a)
{
    return A*(1.0/a);  

}

}
