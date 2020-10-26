#ifndef MSNHCVMAT_H
#define MSNHCVMAT_H

#include <algorithm>
#include "Msnhnet/cv/MsnhCVType.h"
#include "Msnhnet/utils/MsnhException.h"
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/utils/MsnhExString.h"
#include <iostream>
#include <iomanip>
#include <random>

namespace Msnhnet
{
class MsnhNet_API Mat
{
public:

    Mat (const Mat& mat);
    Mat (const std::string &path);
    Mat (const int &width, const int &height, const MatType &matType);
    Mat (const int &width, const int &height, const MatType &matType, void *data);
    Mat ();
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
        else if(fmt=='d')
        {
            memcpy(&val, this->_data.f64+(this->_width*pos.x2 + pos.x1)*array, array*8);
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
            return;

        }

        if(fmt=='b')
        {
            memcpy(this->_data.u8+(this->_width*pos.x2 + pos.x1)*array, &val, array);
        }
        else if(fmt=='f')
        {
            memcpy(this->_data.f32+(this->_width*pos.x2 + pos.x1)*array, &val, array*4);
        }
        else if(fmt=='d')
        {
            memcpy(this->_data.f64+(this->_width*pos.x2 + pos.x1)*array, &val, array*8);
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
                else if(fmt == 'd')
                {
                    memcpy(this->_data.f64+(this->_width*i + j)*array, &val, array*8);
                }
            }
        }
    }

    void checkPixelType(const int &array, const int &fmt);

    void readImage(const std::string& path);

    void saveImage(const std::string& path, const SaveImageType &saveImageType, const int &quality=100);

    void saveImage(const std::string& path, const int &quality=100);

    void release();

    void copyTo(Mat &mat);

    void convertTo(Mat &dst, const CvtDataType& cvtDataType);

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

    bool isEmpty() const; 

    Vec2I32 getSize();

    size_t getDataNum();

    uint8_t getPerDataByteNum();

    /* one channel  */
    static Mat eye(const int &num, const MatType &matType);
    static Mat dense(const int &width, const int &height, const MatType &matType, const float &val);
    static Mat diag(const int &num, const MatType &matType, const float &val);
    static Mat random(const int &width, const int &height, const MatType &matType);
    static Mat randomDiag(const int &num, const MatType &matType);
    template<typename T>
    static T randUniform(T min, T max)
    {
        if(max < min)
        {
            T swap = min;
            min = max;
            max = swap;
        }

    #if (RAND_MAX < 65536)
        int rnd = rand()*(RAND_MAX + 1) + rand();
        return ((T)rnd / (RAND_MAX*RAND_MAX) * (max - min)) + min;
    #else
        return ((T)rand() / RAND_MAX * (max - min)) + min;
    #endif
    }

    Mat transpose();

    double det();

    /*      [ U U U U U ]
     *      [ L U U U U ]
     *  A = [ L L U U U ]
     *      [ L L L U U ]
     *      [ L L L L U ]
     * */
    Mat invert();

    Mat solve();
    /*================*/

    void printMat();

    bool isF32Mat() const; 

    bool isF64Mat() const; 

    bool isU8Mat() const; 

    bool isOneChannel() const; 

    static Mat add(const Mat &A, const Mat &B);

    static Mat sub(const Mat &A, const Mat &B);

    static Mat dot(const Mat &A, const Mat &B);

    static Mat div(const Mat &A, const Mat &B);

    static Mat mul(const Mat &A, const Mat &B);

    Mat &operator = (const Mat &mat);

    friend Mat operator+ (const Mat &A, const Mat &B);
    friend Mat operator+ (const double &a, const Mat &A);
    friend Mat operator+ (const Mat &A, const double &a);

    friend Mat operator- (const Mat &A, const Mat &B);
    friend Mat operator- (const double &a, const Mat &A);
    friend Mat operator- (const Mat &A, const double &a);

    friend Mat operator* (const Mat &A, const Mat &B);
    friend Mat operator* (const double &a, const Mat &A);
    friend Mat operator* (const Mat &A, const double &a);

    friend Mat operator/ (const Mat &A, const Mat &B);
    friend Mat operator/ (const double &a, const Mat &A);
    friend Mat operator/ (const Mat &A, const double &a);
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

