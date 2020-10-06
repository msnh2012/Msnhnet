#ifndef MSNHCVMAT_H
#define MSNHCVMAT_H
#include <algorithm>
#include <Msnhnet/cv/MsnhCVType.h>
#include <Msnhnet/utils/MsnhException.h>
#include <Msnhnet/config/MsnhnetCfg.h>
#include <iostream>

namespace Msnhnet
{
class Mat
{
public:
    Mat ();
    Mat (const Mat& mat);
    Mat (const int &width, const int &height, const MatType &matType);
    Mat (const int &width, const int &height, const MatType &matType, void *data);
    ~Mat ();

    template<typename T>
    T getPixel(const Vec2I32 &pos)
    {
        int array   = DataType<T>::array;
        int fmt     = DataType<T>::fmt;

        checkPixelType(array, fmt);

        if(pos.x1 < 0 || pos.x2 < 0 || pos.x1 >= this->_width || pos.x2>= this->_height)
        {
            throw Exception(1,"[CV]: pixel pos out of memory", __FILE__, __LINE__, __FUNCTION__);
        }

        T val;

        if(fmt=='b')
        {
            memcpy(&val, this->_data.u8+(this->_width*pos.x2 + pos.x1)*array, array);
        }
        else if(fmt=='f')
        {
            memcpy(&val, this->_data.f32+(this->_width*pos.x2 + pos.x1)*array, array*4);
        }

        return val;
    }

    template<typename T>
    void setPixel(const Vec2I32 &pos, const T &val)
    {
        int array   = DataType<T>::array;
        int fmt     = DataType<T>::fmt;

        checkPixelType(array, fmt);

        if(pos.x1 < 0 || pos.x2 < 0 || pos.x1 >= this->_width || pos.x2>= this->_height)
        {
            throw Exception(1,"[CV]: pixel pos out of memory", __FILE__, __LINE__, __FUNCTION__);
        }

        if(fmt=='b')
        {
            memcpy(this->_data.u8+(this->_width*pos.x2 + pos.x1)*array, &val, array);
        }
        else if(fmt=='f')
        {
            memcpy(this->_data.f32+(this->_width*pos.x2 + pos.x1)*array, &val, array*4);
        }
    }

    template<typename T>
    void fillPixel(const T &val)
    {
        int array   = DataType<T>::array;
        int fmt     = DataType<T>::fmt;
        checkPixelType(array, fmt);

        if(this->_width == 0 || this->_height == 0)
        {
            throw Exception(1,"[CV]: width == 0 || height == 0!", __FILE__, __LINE__, __FUNCTION__);
        }

        for (int i = 0; i < this->_height; ++i)
        {
            for (int j = 0; j < this->_width; ++j)
            {
                if(fmt == 'b')
                {
                    memcpy(this->_data.u8+(this->_width*i + j)*array, &val, array);
                }
                else if(fmt == 'f')
                {
                    memcpy(this->_data.f32+(this->_width*i + j)*array, &val, array*4);
                }
            }
        }
    }

    void checkPixelType(const int &array, const int &fmt);

    void readImage(const std::string& path);

    void saveImage(const std::string& path, const SaveImageType &saveImageType, const int &quality=100);

    void clearMat();

    Mat operator + (const Mat & mat);

    Mat operator - (const Mat & mat);

    Mat operator * (const Mat & mat);

    Mat operator / (const Mat & mat);

    Mat &operator = (const Mat & mat);

    void copyTo(Mat &mat);

    int getWidth() const;

    int getHeight() const;

    int getChannel() const;

    int getStep() const;

    MatType getMatType() const;

    MatData getData() const;

    void setWidth(int width);

    void setHeight(int height);

    void setChannel(int channel);

    void setStep(int step);

    void setMatType(const MatType &matType);

    void setU8Ptr(uint8_t *const &ptr);

    bool isEmpty();

private:
    int _width          = 0;
    int _height         = 0;
    int _channel        = 0;
    int _step           = 0;
    MatType _matType    = MatType::MAT_RGB_F32;
    MatData _data;
};
}

#endif 

