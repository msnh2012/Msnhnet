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
#include <tuple>

namespace Msnhnet
{

class MsnhNet_API Mat
{
public:

    Mat (const Mat& mat);
    Mat (const std::string &path);
    Mat (const int &width, const int &height, const MatType &matType, void *data=nullptr);
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
            throw Exception(1,"[Mat]: pixel pos out of memory", __FILE__, __LINE__, __FUNCTION__);
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
    T getPixelAtRowCol(const Vec2I32 &pos)
    {
        return getPixel<T>({pos.x2,pos.x1});
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
            throw Exception(1,"[Mat]: width == 0 || height == 0!", __FILE__, __LINE__, __FUNCTION__);
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

    template<typename T>
    static void createMat(const int &width, const int &height, const int &channel, Mat &mat, T* data=nullptr)
    {
        if(!std::is_same<T,uint8_t>::value && !std::is_same<T,float>::value && !std::is_same<T,double>::value)
        {
            throw Exception(1,"[Mat]: createMat only uint8_t float and double supported!", __FILE__, __LINE__, __FUNCTION__);
        }

        if(std::is_same<T,uint8_t>::value)
        {
            if(channel==1)
            {
                mat = Mat(width,height,MAT_GRAY_U8,data);
            }
            else if(channel==3)
            {
                mat = Mat(width,height,MAT_RGB_U8,data);
            }
            else if(channel==4)
            {
                mat = Mat(width,height,MAT_RGBA_U8,data);
            }
        }
        else if(std::is_same<T,float>::value)
        {
            if(channel==1)
            {
                mat = Mat(width,height,MAT_GRAY_F32,data);
            }
            else if(channel==3)
            {
                mat = Mat(width,height,MAT_RGB_F32,data);
            }
            else if(channel==4)
            {
                mat = Mat(width,height,MAT_RGBA_F32,data);
            }
        }
        else if(std::is_same<T,double>::value)
        {
            if(channel==1)
            {
                mat = Mat(width,height,MAT_GRAY_F64,data);
            }
            else if(channel==3)
            {
                mat = Mat(width,height,MAT_RGB_F64,data);
            }
            else if(channel==4)
            {
                mat = Mat(width,height,MAT_RGBA_F64,data);
            }
        }
    }

    void checkPixelType(const int &array, const int &fmt);

    void readImage(const std::string& path);

    void saveImage(const std::string& path, const SaveImageType &saveImageType, const int &quality=100);

    void saveImage(const std::string& path, const int &quality=100);

    std::vector<char> encodeToMemory(const MatEncodeType &encodeType=MAT_ENCODE_JPG, const int &jpgQuality=100);

    void decodeFromMemory(char *data, const size_t &dataLen);

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

    uint8_t *getBytes();

    float *getFloat32();

    double *getFloat64();

    bool isEmpty() const; 

    Vec2I32 getSize() const;

    size_t getDataNum() const;

    uint8_t getPerDataByteNum() const;

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

    /*      [ 1 0 0 0 0 ]  [ U U U U U ]
     *      [ L 1 0 0 0 ]  [ 0 U U U U ]
     *  A = [ L L 1 0 0 ]  [ 0 0 U U U ]
     *      [ L L L 1 0 ]  [ 0 0 0 U U ]
     *      [ L L L L 1 ]  [ 0 0 0 0 U ]
     * */
    std::vector<Mat> LUDecomp(bool outLU=true);

    /*      [ a 0 0 0 0 ]  [ a b d g k ]
     *      [ b c 0 0 0 ]  [ 0 c e h l ]
     *  A = [ d e f 0 0 ]  [ 0 0 f i m ]
     *      [ g h i j 0 ]  [ 0 0 0 j n ]
     *      [ k l m n o ]  [ 0 0 0 0 o ]
     * */
    std::vector<Mat> CholeskyDeComp(bool outChols=true);

    Mat invert(const DecompType &decompType=DECOMP_LU);

    Mat solve();
    /*================*/

    void print();

    bool isF32Mat() const; 

    bool isF64Mat() const; 

    bool isU8Mat() const; 

    bool isOneChannel() const; 

    static Mat add(const Mat &A, const Mat &B);

    static Mat sub(const Mat &A, const Mat &B);

    static Mat mul(const Mat &A, const Mat &B);

    static Mat div(const Mat &A, const Mat &B);

    static Mat dot(const Mat &A, const Mat &B);

    Mat &operator = (const Mat &mat);

    MsnhNet_API friend Mat operator+ (const Mat &A, const Mat &B);
    MsnhNet_API friend Mat operator+ (const double &a, const Mat &A);
    MsnhNet_API friend Mat operator+ (const Mat &A, const double &a);

    MsnhNet_API friend Mat operator- (const Mat &A, const Mat &B);
    MsnhNet_API friend Mat operator- (const double &a, const Mat &A);
    MsnhNet_API friend Mat operator- (const Mat &A, const double &a);

    MsnhNet_API friend Mat operator* (const Mat &A, const Mat &B);
    MsnhNet_API friend Mat operator* (const double &a, const Mat &A);
    MsnhNet_API friend Mat operator* (const Mat &A, const double &a);

    MsnhNet_API friend Mat operator/ (const Mat &A, const Mat &B);
    MsnhNet_API friend Mat operator/ (const double &a, const Mat &A);
    MsnhNet_API friend Mat operator/ (const Mat &A, const double &a);
protected:

    int _width          = 0;
    int _height         = 0;
    int _channel        = 0;
    int _step           = 0;
    MatType _matType    = MatType::MAT_RGB_F32;
    MatData _data;
};
void bufferFromCallback(void* context, void* data, int size);

template<int w,int h,typename T>
class MsnhNet_API Mat_:public Mat
{
public:
    Mat_():Mat(w,h,std::is_same<T,double>::value?MAT_GRAY_F64:(std::is_same<T,uint8_t>::value?MAT_GRAY_U8:MAT_GRAY_F32))
    {

    }

    void setVal(const std::vector<T> &val)
    {
        if(val.size()!=this->getDataNum())
        {
            throw Exception(1,"[Mat_]: set val num must equal mat data num! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        memcpy(this->getBytes(), val.data(),val.size()*sizeof(T));

    }

    T getVal(const uint8_t &index)
    {
        if(index>getDataNum()-1)
        {
            throw Exception(1,"[Mat_]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        return *((T*)this->getBytes()+index);
    }

    T getVal(const uint8_t& width, const uint8_t& height)
    {
        uint16_t index = height*this->_width + width;
        if(index > getDataNum()-1)
        {
            throw Exception(1,"[Mat_]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        return *((T*)this->getBytes()+index);
    }

    T getValAtRowCol(const uint8_t& row, const uint8_t& col)
    {
        uint16_t index = row*this->_width + col;
        if(index > getDataNum()-1)
        {
            throw Exception(1,"[Mat_]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        return *((T*)this->getBytes()+index);
    }

    Mat_(Mat &mat)
    {
        release();
        this->_channel  = mat.getChannel();
        this->_width    = mat.getWidth();
        this->_height   = mat.getHeight();
        this->_step     = mat.getStep();
        this->_matType  = mat.getMatType();

        if(mat.getBytes()!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
    }

    Mat_(Mat mat)
    {
        release();
        this->_channel  = mat.getChannel();
        this->_width    = mat.getWidth();
        this->_height   = mat.getHeight();
        this->_step     = mat.getStep();
        this->_matType  = mat.getMatType();

        if(mat.getBytes()!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
    }

    Mat_& operator= (Mat_ &mat)
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

    Mat_& operator= (Mat &mat)
    {
        release();
        this->_channel  = mat.getChannel();
        this->_width    = mat.getWidth();
        this->_height   = mat.getHeight();
        this->_step     = mat.getStep();
        this->_matType  = mat.getMatType();

        if(mat.getBytes()!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
        return *this;
    }

    Mat_& operator= (Mat mat)
    {
        release();
        this->_channel  = mat.getChannel();
        this->_width    = mat.getWidth();
        this->_height   = mat.getHeight();
        this->_step     = mat.getStep();
        this->_matType  = mat.getMatType();

        if(mat.getBytes()!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
        return *this;
    }
};

class MsnhNet_API Quaternion
{
public:
    Quaternion(){}
    Quaternion(const double& q0, const double& q1, const double& q2, const double& q3);

    void setVal(std::vector<double> &val);

    std::vector<double> getVal();

    double mod();

    Quaternion invert();

    double getQ0() const;
    double getQ1() const;
    double getQ2() const;
    double getQ3() const;

    void print();

    double operator[] (const uint8_t& index);

    MsnhNet_API friend Quaternion operator- (const Quaternion &A, const Quaternion &B);
    MsnhNet_API friend Quaternion operator+ (const Quaternion &A, const Quaternion &B);
    MsnhNet_API friend Quaternion operator* (const Quaternion &A, const Quaternion &B);
    MsnhNet_API friend Quaternion operator/ (const Quaternion &A, Quaternion &B);
private:
    double _q0 = 0;
    double _q1 = 0;
    double _q2 = 0;
    double _q3 = 0;
};

typedef Mat_<3,3,double> RotationMat;
typedef Mat_<4,1,double> Quat;
typedef Mat_<3,1,double> Euler;
typedef Mat_<3,1,double> RotationVec;

typedef Mat_<2,2,uint8_t> Mat2x2B;
typedef Mat_<2,2,float> Mat2x2F;
typedef Mat_<2,2,double> Mat2x2D;

typedef Mat_<3,3,uint8_t> Mat3x3B;
typedef Mat_<3,3,float> Mat3x3F;
typedef Mat_<3,3,double> Mat3x3D;

typedef Mat_<4,4,uint8_t> Mat4x4B;
typedef Mat_<4,4,float> Mat4x4F;
typedef Mat_<4,4,double> Mat4x4D;

typedef Mat_<2,1,uint8_t> Mat1x2B;
typedef Mat_<2,1,float> Mat1x2F;
typedef Mat_<2,1,double> Mat1x2D;

typedef Mat_<3,1,uint8_t> Mat1x3B;
typedef Mat_<3,1,float> Mat1x3F;
typedef Mat_<3,1,double> Mat1x3D;

typedef Mat_<4,1,uint8_t> Mat1x4B;
typedef Mat_<4,1,float> Mat1x4F;
typedef Mat_<4,1,double> Mat1x4D;

typedef Mat_<1,2,uint8_t> Mat2x1B;
typedef Mat_<1,2,float> Mat2x1F;
typedef Mat_<1,2,double> Mat2x1D;

typedef Mat_<1,3,uint8_t> Mat3x1B;
typedef Mat_<1,3,float> Mat3x1F;
typedef Mat_<1,3,double> Mat3x1D;

typedef Mat_<1,4,uint8_t> Mat4x1B;
typedef Mat_<1,4,float> Mat4x1F;
typedef Mat_<1,4,double> Mat4x1D;

typedef Mat_<2,3,uint8_t> Mat3x2B;
typedef Mat_<2,3,float> Mat3x2F;
typedef Mat_<2,3,double> Mat3x2D;

typedef Mat_<3,2,uint8_t> Mat2x3B;
typedef Mat_<3,2,float> Mat2x3F;
typedef Mat_<3,2,double> Mat2x3D;

typedef Mat_<2,4,uint8_t> Mat4x2B;
typedef Mat_<2,4,float> Mat4x2F;
typedef Mat_<2,4,double> Mat4x2D;

typedef Mat_<4,2,uint8_t> Mat2x4B;
typedef Mat_<4,2,float> Mat2x4F;
typedef Mat_<4,2,double> Mat2x4D;

typedef Mat_<3,4,uint8_t> Mat4x3B;
typedef Mat_<3,4,float> Mat4x3F;
typedef Mat_<3,4,double> Mat4x3D;

typedef Mat_<4,3,uint8_t> Mat3x4B;
typedef Mat_<4,3,float> Mat3x4F;
typedef Mat_<4,3,double> Mat3x4D;

}

#endif 

