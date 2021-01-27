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
#define MSNH_F32_EPS 1E-6
#define MSNH_F64_EPS 1E-15
class MsnhNet_API Mat
{
public:

    Mat (const Mat& mat);
    Mat (const std::string &path);  

    Mat (const int &width, const int &height, const MatType &matType, void *data=nullptr);
    Mat ();
    ~Mat ();

    template<typename T>
    T getPixel(const Vec2I32 &pos) const
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

    template<typename T>
    Mat getCol_(const int& col)
    {
        if(col <0)
        {
            throw Exception(1, "[Mat] col should > 0" , __FILE__, __LINE__,__FUNCTION__);
        }

        if(col >= this->_width)
        {
            throw Exception(1, "[Mat] col should < width col-width:(" + std::to_string(col) + ":" + std::to_string(this->_width) + ")" , __FILE__, __LINE__,__FUNCTION__);
        }
        Mat mCol(1,this->_height,this->_matType);
        for (int i = 0; i < this->_height; ++i)
        {
            T val = this->getPixel<T>({col,i});
            mCol.setPixel<T>({0, i},val);
        }
        return mCol;
    }

    template<typename T>
    void setCol_(const int& col, const Mat& mat)
    {
        if(col <0)
        {
            throw Exception(1, "[Mat] col should > 0" , __FILE__, __LINE__,__FUNCTION__);
        }

        if(col >= this->_width)
        {
            throw Exception(1, "[Mat] col should < width col-width:(" + std::to_string(col) + ":" + std::to_string(this->_width) + ")" , __FILE__, __LINE__,__FUNCTION__);
        }

        if(mat.getHeight()!=1 || mat.getWidth()!=this->_height)
        {
            throw Exception(1, "[Mat] input height should == 1 && input width should equal mat height.  in.height-height:(" + std::to_string(mat.getHeight()) + ":" + std::to_string(this->_height) + ")" , __FILE__, __LINE__,__FUNCTION__);
        }

        for (int i = 0; i < this->_height; ++i)
        {
            T val = mat.getPixel<T>({i,0});
            this->setPixel<T>({col,i},val);
        }
    }

    template<typename T>
    Mat getRow_(const int& row)
    {
        if(row <0)
        {
            throw Exception(1, "[Mat] row should > 0" , __FILE__, __LINE__,__FUNCTION__);
        }

        if(row >= this->_height)
        {
            throw Exception(1, "[Mat] row should < height row-height:(" + std::to_string(row) + ":" + std::to_string(this->_height) + ")" , __FILE__, __LINE__,__FUNCTION__);
        }

        Mat mRow(this->_width,1,this->_matType);

        for (int i = 0; i < this->_width; ++i)
        {
            T val = this->getPixel<T>({i,row});
            mRow.setPixel<T>({i,0},val);
        }
        return mRow;
    }

    template<typename T>
    void setRow_(const int& row, const Mat& mat)
    {
        if(row <0)
        {
            throw Exception(1, "[Mat] row should > 0" , __FILE__, __LINE__,__FUNCTION__);
        }

        if(row >= this->_height)
        {
            throw Exception(1, "[Mat] row should < height row-height:(" + std::to_string(row) + ":" + std::to_string(this->_height) + ")" , __FILE__, __LINE__,__FUNCTION__);
        }

        if(mat.getHeight()!=1 || mat.getWidth()!=this->_width)
        {
            throw Exception(1, "[Mat] input height should == 1 && input width should equal mat width.  in.width-width:(" + std::to_string(mat.getWidth()) + ":" + std::to_string(this->_width) + ")" , __FILE__, __LINE__,__FUNCTION__);
        }

        for (int i = 0; i < this->_height; ++i)
        {
            T val = mat.getPixel<T>({i,0});
            this->setPixel<T>({i,row},val);
        }
    }

    void checkPixelType(const int &array, const int &fmt) const;

    void readImage(const std::string& path);

    void saveImage(const std::string& path, const SaveImageType &saveImageType, const int &quality=100);

    void saveImage(const std::string& path, const int &quality=100);

    std::vector<char> encodeToMemory(const MatEncodeType &encodeType=MAT_ENCODE_JPG, const int &jpgQuality=100);

    void decodeFromMemory(char *data, const size_t &dataLen);

    void release();

    void copyTo(Mat &mat);

    void convertTo(Mat &dst, const CvtDataType& cvtDataType);

    Mat toFloat32();

    Mat toFloat64();

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

    uint8_t *getBytes() const;

    float *getFloat32() const;

    double *getFloat64() const;

    double getVal2Double(const size_t &index) const;

    bool isEmpty() const;  

    Vec2I32 getSize() const;

    size_t getDataNum() const;

    size_t getByteNum() const;

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
    std::vector<Mat> LUDecomp(bool outLU=true) const;

    /*      [ a 0 0 0 0 ]  [ a b d g k ]
     *      [ b c 0 0 0 ]  [ 0 c e h l ]
     *  A = [ d e f 0 0 ]  [ 0 0 f i m ]
     *      [ g h i j 0 ]  [ 0 0 0 j n ]
     *      [ k l m n o ]  [ 0 0 0 0 o ]
     * */
    std::vector<Mat> CholeskyDeComp(bool outChols=true) const;

    Mat invert(const DecompType &decompType=DECOMP_LU) const;

    Mat solve();
    /*================*/

    void print();

    std::string toString();

    std::string getMatTypeStr();

    bool isF32Mat() const;  

    bool isF64Mat() const;  

    bool isU8Mat() const;  

    bool isOneChannel() const;  

    /* ================ */
    bool isVector() const;  

    bool isNum() const; 

    bool isMatrix() const;

    bool isVector2D() const;

    bool isVector3D() const;

    bool isVector4D() const;

    bool isMatrix3x3() const;

    bool isMatrix4x4() const;

    static Mat add(const Mat &A, const Mat &B);

    static Mat sub(const Mat &A, const Mat &B);

    static Mat mul(const Mat &A, const Mat &B);

    static Mat div(const Mat &A, const Mat &B);

    static Mat eleWiseDiv(const Mat &A, const Mat &B);

    static Mat eleWiseMul(const Mat &A, const Mat &B);

    bool isNull() const;

    bool isFuzzyNull() const;

    bool isNan() const;

    Mat &operator= (const Mat &mat);

    MsnhNet_API friend bool operator== (const Mat &A, const Mat &B);
    MsnhNet_API friend bool operator!= (const Mat &A, const Mat &B);

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
    Mat_():Mat(w,h,getMatTypeFromT())
    {

    }

    Mat_(const std::vector<T> &val):Mat(w,h,getMatTypeFromT())
    {
        this->setVal(val);
    }

   static MatType getMatTypeFromT()
   {
        if(std::is_same<T,double>::value)
        {
            return MAT_GRAY_F64;
        }
        else if(std::is_same<T,float>::value)
        {
            return MAT_GRAY_F32;
        }
        else if(std::is_same<T,uint8_t>::value)
        {
            return MAT_GRAY_U8;
        }
        else
        {
            throw Exception(1,"[Mat_]: only u8/f32/f64 is supported! \n", __FILE__, __LINE__, __FUNCTION__);
        }
   }

    void setVal(const std::vector<T> &val)
    {
        if(val.size()!=this->getDataNum())
        {
            throw Exception(1,"[Mat_]: set val num must equal mat data num! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        memcpy(this->getBytes(), val.data(),val.size()*sizeof(T));
    }

    void setVal(const size_t &index, const T &val)
    {
        if(index>getDataNum()-1)
        {
            throw Exception(1,"[Mat_]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        *((T*)this->getBytes()+index) = val;
    }

    T getVal(const size_t &index) const
    {
        if(index>getDataNum()-1)
        {
            throw Exception(1,"[Mat_]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        return *((T*)this->getBytes()+index);
    }

    T getVal(const int& width, const int& height) const
    {
        size_t index = height*this->_width + width;
        if(index > getDataNum()-1)
        {
            throw Exception(1,"[Mat_]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        return *((T*)this->getBytes()+index);
    }

    T getValAtRowCol(const int& row, const int& col) const
    {
        size_t index = row*this->_width + col;
        if(index > getDataNum()-1)
        {
            throw Exception(1,"[Mat_]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        return *((T*)this->getBytes()+index);
    }

    void setValAtRowCol(const int& row, const int& col, const T& val)
    {
        size_t index = row*this->_width + col;
        if(index > getDataNum()-1)
        {
            throw Exception(1,"[Mat_]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        *((T*)this->getBytes()+index) = val;
    }

    Mat_(const Mat_ &mat) 

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

    Mat_(const Mat &mat)  

    {
        if(mat.getWidth()!=w || mat.getHeight()!=h || mat.getChannel()!=1 || mat.getMatType()!=getMatTypeFromT())
        {
            throw Exception(1, "[Mat_] mat props should be equal." , __FILE__, __LINE__,__FUNCTION__);
        }

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

    Mat_& operator= (const Mat_ &mat)
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

    Mat_& operator= (const Mat &mat)
    {
        if(mat.getWidth()!=w || mat.getWidth()!=h || mat.getChannel()!=1 || mat.getMatType()!=getMatTypeFromT())
        {
            throw Exception(1, "[Mat_] mat props should be equal." , __FILE__, __LINE__,__FUNCTION__);
        }
        if(this!=&mat)
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
        return *this;
    }

    Mat_ getCol(const int &col) const
    {
        return this->getCol_<T>(col);
    }

    Mat_ getRow(const int &row) const
    {
        return this->getRow_<T>(row);
    }

    void setCol(const int &col, const Mat_<h,1,T> &mat)
    {
        this->setCol_<T>(col,mat);
    }

    void setRow(const int &row, const Mat_<w,1,T> &mat)
    {
        this->setRow_<T>(row,mat);
    }

    static Mat_ eye()
    {
        if(w==h)
        {
            return Mat::eye(w,getMatTypeFromT());
        }
        else
        {
            throw Exception(1, "[Mat_] w!=h no eye matrix." , __FILE__, __LINE__,__FUNCTION__);
        }
    }

};

template<int w, typename T>
class MsnhNet_API Vector :public Mat_<w,1,T>
{
public:

    Vector():Mat_<w,1,T>()
    {

    }

    Vector(const std::vector<T> &val):Mat_<w,1,T>(val)
    {

    }

    Vector(const Vector &vec)

    {
        this->release();
        this->_channel  = vec.getChannel();
        this->_width    = vec.getWidth();
        this->_height   = vec.getHeight();
        this->_step     = vec.getStep();
        this->_matType  = vec.getMatType();

        if(vec.getBytes()!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, vec.getBytes(), this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
    }

    Vector(const Mat &mat) 

    {
        if(mat.getWidth()!=w || mat.getWidth()!=1 || mat.getChannel()!=1 || mat.getMatType()!=getMatTypeFromT())
        {
            throw Exception(1, "[Vector] vector props should be equal." , __FILE__, __LINE__,__FUNCTION__);
        }

        this->release();
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

    Vector& operator= (const Vector &vec)
    {
        if(this!=&vec)
        {
            this->release();
            this->_channel  = vec._channel;
            this->_width    = vec._width;
            this->_height   = vec._height;
            this->_step     = vec._step;
            this->_matType  = vec._matType;

            if(vec._data.u8!=nullptr)
            {
                uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
                memcpy(u8Ptr, vec._data.u8, this->_width*this->_height*this->_step);
                this->_data.u8 =u8Ptr;
            }
        }
        return *this;
    }

    Vector& operator= (const Mat &mat)
    {
        if(mat.getWidth()!=w || mat.getWidth()!=1 || mat.getChannel()!=1 || mat.getMatType()!=getMatTypeFromT())
        {
            throw Exception(1, "[Vector] vector props should be equal." , __FILE__, __LINE__,__FUNCTION__);
        }
        if(this!=&mat)
        {
            this->release();
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
        return *this;
    }

    T operator[] (const int &index) const
    {
        return this->getVal(index);
    }

    void normalize()
    {
        if(this->isU8Mat())
        {
            throw Exception(1, "[Vector] u8 normalize is not supported!", __FILE__, __LINE__,__FUNCTION__);
        }
        else
        {
            if(this->isF32Mat())
            {
                float len = 0;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:len)
#endif
                for (int i = 0; i < w; ++i)
                {
                    len += this->_data.f32[i]*this->_data.f32[i];
                }

                if(fabsf(len - 1.0f) < MSNH_F32_EPS || fabsf(len) < MSNH_F32_EPS)
                {
                    return;
                }

                len = sqrtf(len);

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
                for (int i = 0; i < w; ++i)
                {
                    this->_data.f32[i] = this->_data.f32[i] / len;
                }
            }
            else if(this->isF64Mat())
            {
                double len = 0;

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:len)
#endif
                for (int i = 0; i < w; ++i)
                {
                    len += this->_data.f64[i]*this->_data.f64[i];
                }

                if(std::fabs(len - 1.0) < MSNH_F64_EPS || std::fabs(len) < MSNH_F64_EPS)
                {
                    return;
                }

                len = std::sqrt(len);

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
                for (int i = 0; i < w; ++i)
                {
                    this->_data.f64[i] = this->_data.f64[i]/len;
                }
            }
        }
    }

    Vector normalized() const
    {
        if(this->isU8Mat())
        {
            throw Exception(1, "[Vector] u8 normalize is not supported!", __FILE__, __LINE__,__FUNCTION__);
        }
        else
        {
            if(this->isF32Mat())
            {
                float len = 0;

                Vector vec;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:len)
#endif
                for (int i = 0; i < w; ++i)
                {
                    len += this->_data.f32[i]*this->_data.f32[i];
                }

                if(fabsf(len - 1.0f) < MSNH_F32_EPS)
                {
                    return *this;
                }

                if(fabsf(len) < MSNH_F32_EPS)
                {
                    return vec;
                }

                len = sqrtf(len);

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
                for (int i = 0; i < w; ++i)
                {
                    vec.getData().f32[i] = this->_data.f32[i] / len;
                }

                return vec;
            }
            else if(this->isF64Mat())
            {
                double len = 0;

                Vector vec;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:len)
#endif
                for (int i = 0; i < w; ++i)
                {
                    len += this->_data.f64[i]*this->_data.f64[i];
                }

                if(std::fabs(len - 1.0) < MSNH_F64_EPS)
                {
                    return *this;
                }

                if(std::fabs(len) < MSNH_F64_EPS)
                {
                    return vec;
                }

                len = std::sqrt(len);

#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
                for (int i = 0; i < w; ++i)
                {
                    vec.getData().f64[i] = this->_data.f64[i]/len;
                }

                return vec;
            }
        }
    }

    inline double length() const
    {
        double len = 0;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:len)
#endif
        for (int i = 0; i < w; ++i)
        {
            len += this->getVal(i)*this->getVal(i);
        }
        return std::sqrt(len);
    }

    inline double lengthSquared() const
    {
        double len = 0;
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD) reduction(+:len)
#endif
        for (int i = 0; i < w; ++i)
        {
            len += this->getVal(i)*this->getVal(i);
        }
        return len;
    }

    /* 点到点之间的距离
     * .eg ^
     *     |
     *   A x       --> -->    --->
     *     | \     OA - OB  = |BA|
     *     |   \
     *   O |-----x-->
     *           B
     */
    inline double distanceToPoint(const Vector& point) const
    {
        return Vector(*this - point).length();
    }

    /* 点到线之间的距离
     * .eg ^
     *   \ |
     *     x      x(A)
     *     | \
     *     |   x (point)
     *     |     \
     *   O |-------x--> B
     *               \(direction)
     *                 \LINE(point + direction)
     */
    inline double distanceToLine(const Vector& point, const Vector& direction) const
    {
        if(direction.getWidth()<2 || point.getWidth()<2)
        {
            throw Exception(1,"[Vector] only 2 dims+ is supported!",__FILE__,__LINE__,__FUNCTION__);
        }

        if(direction.isFuzzyNull())
        {
            return Vector(*this - point).length();
        }

        Vector p = point + Vector::eleWiseMul((*this - point)*direction,direction);
        return (*this - p).length();
    }

    /*          点到线之间的距离
     *          .eg ^
     *          / \ |      *(normal)
     *         /    x    *
     *        /     | \
     *       /      |   \    x(A)
     *       \     *|     \
     *         \  O |-------x--> B
     *         * \ /       /
     *        *   /\      /
     *           /   \   / (plane)
     *          /      \/
     *
     */
    inline double distanceToPlane(const Vector& plane, const Vector& normal) const
    {
        if(plane.getWidth()<3 || normal.getWidth()<3)
        {
            throw Exception(1,"[Vector] only 3 dims+ is supported!",__FILE__,__LINE__,__FUNCTION__);
        }

        return (*this-plane)*normal;
    }
};

typedef Mat_<3,3,double> RotationMatD;
typedef Mat_<3,3,double> Matrix3x3D;
typedef Vector<3,double> EulerD;
typedef Vector<3,double> TransformD;
typedef Vector<3,double> RotationVecD;
typedef Vector<2,double> Vector2D;
typedef Vector<4,double> Vector4D;

class MsnhNet_API Vector3D : public Vector<3,double>
{
public:
    Vector3D(){}

    Vector3D(const std::vector<double> &val):Vector<3,double>(val){}

    Vector3D(const Vector3D &vec);

    Vector3D(const Mat &mat);  

    Vector3D& operator= (const Vector3D &vec);

    Vector3D& operator= (const Mat &mat);

    static Vector3D crossProduct(const Vector3D &v1, const Vector3D &v2);

    static Vector3D normal(const Vector3D &v1, const Vector3D &v2);

    static Vector3D normal(const Vector3D &v1, const Vector3D &v2, const Vector3D &v3);

};

class MsnhNet_API QuaternionD
{
public:
    QuaternionD(){}
    QuaternionD(const QuaternionD &q);
    QuaternionD(const double& q0, const double& q1, const double& q2, const double& q3);
    QuaternionD(const std::vector<double> &val);

    void setVal(const std::vector<double> &val);

    std::vector<double> getVal() const;

    double mod() const;

    QuaternionD invert() const;

    double getQ0() const;
    double getQ1() const;
    double getQ2() const;
    double getQ3() const;

    void print();

    std::string toString();

    double operator[] (const uint8_t& index);

    QuaternionD& operator=(const QuaternionD& q);

    bool operator== (const QuaternionD& q);

    MsnhNet_API friend QuaternionD operator- (const QuaternionD &A, const QuaternionD &B);
    MsnhNet_API friend QuaternionD operator+ (const QuaternionD &A, const QuaternionD &B);
    MsnhNet_API friend QuaternionD operator* (const QuaternionD &A, const QuaternionD &B);
    MsnhNet_API friend QuaternionD operator/ (const QuaternionD &A, const QuaternionD &B);
private:
    double _q0 = 0;
    double _q1 = 0;
    double _q2 = 0;
    double _q3 = 0;
};

class MsnhNet_API QuaternionF
{
public:
    QuaternionF(){}
    QuaternionF(const QuaternionF &q);
    QuaternionF(const float& q0, const float& q1, const float& q2, const float& q3);
    QuaternionF(const std::vector<float> &val);

    void setVal(const std::vector<float> &val);

    std::vector<float> getVal() const;

    float mod() const;

    QuaternionF invert() const;

    void print();

    std::string toString();

    float operator[] (const uint8_t& index);

    QuaternionF& operator=(const QuaternionF& q);

    bool operator ==(const QuaternionF& q);

    MsnhNet_API friend QuaternionF operator- (const QuaternionF &A, const QuaternionF &B);
    MsnhNet_API friend QuaternionF operator+ (const QuaternionF &A, const QuaternionF &B);
    MsnhNet_API friend QuaternionF operator* (const QuaternionF &A, const QuaternionF &B);
    MsnhNet_API friend QuaternionF operator/ (const QuaternionF &A, const QuaternionF &B);

    float getQ0() const;
    float getQ1() const;
    float getQ2() const;
    float getQ3() const;

private:
    float _q0 = 0;
    float _q1 = 0;
    float _q2 = 0;
    float _q3 = 0;
};

typedef Mat_<3,3,float> RotationMatF;
typedef Mat_<3,3,float> Matrix3x3F;
typedef Vector<3,float> EulerF;
typedef Vector<3,float> TransformF;
typedef Vector<3,float> RotationVecF;
typedef Vector<2,float> Vector2F;
typedef Vector<4,float> Vector4F;

class MsnhNet_API Vector3F : public Vector<3,float>
{
public:
    Vector3F(){}

    Vector3F(const std::vector<float> &val):Vector<3,float>(val){}

    Vector3F(const Vector3F &vec);

    Vector3F(const Mat &mat);  

    Vector3F& operator= (const Vector3F &vec);

    Vector3F& operator= (const Mat &mat);

    static Vector3F crossProduct(const Vector3F &v1, const Vector3F &v2);

    static Vector3F normal(const Vector3F &v1, const Vector3F &v2);

    static Vector3F normal(const Vector3F &v1, const Vector3F &v2, const Vector3F &v3);

};
}

#endif 

