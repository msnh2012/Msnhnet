
#include "Msnhnet/cv/MsnhCVMat.h"

#define  STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define  STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace Msnhnet
{

Mat::Mat()
{
}

Mat::Mat(const Mat &mat) 

{
    release();
    this->_channel  = mat.getChannel();
    this->_height   = mat.getHeight();
    this->_width    = mat.getWidth();
    this->_matType  = mat.getMatType();
    this->_step     = mat.getStep();

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
    case MatType::MAT_RGB_U8:
        this->_channel  = 3;
        this->_step     = 3;
        break;
    case MatType::MAT_RGB_F32:
        this->_channel  = 3;
        this->_step     = 12;
        break;
    case MatType::MAT_RGBA_U8:
        this->_channel  = 4;
        this->_step     = 4;
        break;
    case MatType::MAT_RGBA_F32:
        this->_channel  = 4;
        this->_step     = 16;
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
    case MatType::MAT_RGB_U8:
        this->_channel  = 3;
        this->_step     = 3;
        break;
    case MatType::MAT_RGB_F32:
        this->_channel  = 3;
        this->_step     = 12;
        break;
    case MatType::MAT_RGBA_U8:
        this->_channel  = 4;
        this->_step     = 4;
        break;
    case MatType::MAT_RGBA_F32:
        this->_channel  = 4;
        this->_step     = 16;
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
    if(fmt!='b' && fmt!='f' )
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

    if(this->_data.u8==nullptr || this->_width ==0 || this->_height==0)
    {
        throw Exception(1,"[CV]: img empty!", __FILE__, __LINE__, __FUNCTION__);
    }

    int ret;
    switch (saveImageType)
    {
    case SaveImageType::MAT_SAVE_BMP:
        ret = stbi_write_bmp(path.c_str(), this->_width, this->_height,this->_channel,this->_data.u8);
        break;
    case SaveImageType::MAT_SAVE_JPG:
        ret = stbi_write_jpg(path.c_str(), this->_width, this->_height,this->_channel,this->_data.u8, quality);
        break;
    case SaveImageType::MAT_SAVE_PNG:
        ret = stbi_write_png(path.c_str(), this->_width, this->_height,this->_channel,this->_data.u8,0);
        break;
    case SaveImageType::MAT_SAVE_HDR:
        ret = stbi_write_hdr(path.c_str(), this->_width, this->_height,this->_channel,this->_data.f32);
        break;
    case SaveImageType::MAT_SAVE_TGA:
        ret = stbi_write_tga(path.c_str(), this->_width, this->_height,this->_channel,this->_data.u8);
        break;
    }

    if(ret<1)
    {
        throw Exception(1,"[CV]: save image error!", __FILE__, __LINE__, __FUNCTION__);
    }

}

void Mat::saveImage(const std::string &path, const int &quality)
{
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
    this->_matType  = MatType::MAT_RGB_F32;
}

Mat Mat::operator + (const Mat &mat)
{
    Mat tmpMat;

    if(mat.getMatType() != this->_matType || mat.getChannel() != this->_channel || mat.getStep() != this->_step ||
            mat.getWidth() != this->_width || mat.getHeight() != this->_height)
    {
        throw Exception(1,"[CV]: mat properties not equal!", __FILE__, __LINE__, __FUNCTION__);
    }

    tmpMat = mat;

    if(this->_matType == MAT_GRAY_U8 || this->_matType == MAT_RGB_U8 || this->_matType == MAT_RGBA_U8)
    {
        int datas = this->_width*this->_height*this->_step;
        for (int i = 0; i < datas; ++i)
        {
            int add = this->_data.u8[i] + tmpMat.getData().u8[i];

            add = (add>255)?255:add;

            tmpMat.getData().u8[i] = static_cast<uint8_t>(add);

        }
    }
    else if(this->_matType == MAT_GRAY_F32 || this->_matType == MAT_RGB_F32 || this->_matType == MAT_RGBA_F32)
    {
        int datas = this->_width*this->_height*this->_step;
        for (int i = 0; i < datas; ++i)
        {
            tmpMat.getData().f32[i] = tmpMat.getData().f32[i]+this->_data.f32[i];
        }
    }
    return tmpMat;
}

Mat Mat::operator - (const Mat &mat)
{
    Mat tmpMat;

    if(mat.getMatType() != this->_matType || mat.getChannel() != this->_channel || mat.getStep() != this->_step ||
            mat.getWidth() != this->_width || mat.getHeight() != this->_height)
    {
        throw Exception(1,"[CV]: mat properties not equal!", __FILE__, __LINE__, __FUNCTION__);
    }

    tmpMat = mat;

    if(this->_matType == MAT_GRAY_U8 || this->_matType == MAT_RGB_U8 || this->_matType == MAT_RGBA_U8)
    {
        int datas = this->_width*this->_height*this->_step;
        for (int i = 0; i < datas; ++i)
        {
            int sub = this->_data.u8[i] - tmpMat.getData().u8[i];

            sub = (sub<0)?0:sub;

            tmpMat.getData().u8[i] = static_cast<uint8_t>(sub);

        }
    }
    else if(this->_matType == MAT_GRAY_F32 || this->_matType == MAT_RGB_F32 || this->_matType == MAT_RGBA_F32)
    {
        int datas = this->_width*this->_height*this->_step;
        for (int i = 0; i < datas; ++i)
        {
            tmpMat.getData().f32[i] = tmpMat.getData().f32[i]-this->_data.f32[i];
        }
    }
    return tmpMat;
}

Mat Mat::operator * (const Mat &mat)
{
    Mat tmpMat;

    if(mat.getMatType() != this->_matType || mat.getChannel() != this->_channel || mat.getStep() != this->_step ||
            mat.getWidth() != this->_width || mat.getHeight() != this->_height)
    {
        throw Exception(1,"[CV]: mat properties not equal!", __FILE__, __LINE__, __FUNCTION__);
    }

    tmpMat = mat;

    if(this->_matType == MAT_GRAY_U8 || this->_matType == MAT_RGB_U8 || this->_matType == MAT_RGBA_U8)
    {
        int datas = this->_width*this->_height*this->_step;
        for (int i = 0; i < datas; ++i)
        {
            int mul = this->_data.u8[i] * tmpMat.getData().u8[i];

            mul = (mul>255)?255:mul;

            tmpMat.getData().u8[i] = static_cast<uint8_t>(mul);

        }
    }
    else if(this->_matType == MAT_GRAY_F32 || this->_matType == MAT_RGB_F32 || this->_matType == MAT_RGBA_F32)
    {
        int datas = this->_width*this->_height*this->_step;
        for (int i = 0; i < datas; ++i)
        {
            tmpMat.getData().f32[i] = tmpMat.getData().f32[i]*this->_data.f32[i];
        }
    }
    return tmpMat;
}

Mat Mat::operator / (const Mat &mat)
{
    Mat tmpMat;

    if(mat.getMatType() != this->_matType || mat.getChannel() != this->_channel || mat.getStep() != this->_step ||
            mat.getWidth() != this->_width || mat.getHeight() != this->_height)
    {
        throw Exception(1,"[CV]: mat properties not equal!", __FILE__, __LINE__, __FUNCTION__);
    }

    tmpMat = mat;

    if(this->_matType == MAT_GRAY_U8 || this->_matType == MAT_RGB_U8 || this->_matType == MAT_RGBA_U8)
    {
        int datas = this->_width*this->_height*this->_step;
        for (int i = 0; i < datas; ++i)
        {
            int div = 0;
            if(tmpMat.getData().u8[i] == 0)
            {
                div = 255;
            }
            else
            {
                div = this->_data.u8[i] / (tmpMat.getData().u8[i]);
            }

            tmpMat.getData().u8[i] = static_cast<uint8_t>(div);

        }
    }
    else if(this->_matType == MAT_GRAY_F32 || this->_matType == MAT_RGB_F32 || this->_matType == MAT_RGBA_F32)
    {
        int datas = this->_width*this->_height*this->_step;
        for (int i = 0; i < datas; ++i)
        {
            tmpMat.getData().f32[i] = tmpMat.getData().f32[i]/this->_data.f32[i];
        }
    }
    return tmpMat;
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
            memcpy(u8Ptr, mat.getData().u8, this->_width*this->_height*this->_step);
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

bool Mat::isEmpty()
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

uint8_t Mat::getPerDataByteNum()
{
    return static_cast<uint8_t>(this->_step/this->_channel);
}

}
