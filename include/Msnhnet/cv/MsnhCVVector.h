#ifndef MSNHCVVECTOR_H
#define MSNHCVVECTOR_H

#include <Msnhnet/config/MsnhnetCfg.h>
#include <Msnhnet/cv/MsnhCVType.h>
#include <iostream>
#include <sstream>
#include <iomanip>
namespace Msnhnet
{

template<typename T>
class VectorX
{
public:
    VectorX(const int &n)
    {
        _n = n;
        _value = new T[n]();
    }

    VectorX(){}

    ~VectorX()
    {
        release();
    }

    VectorX(const std::vector<T> &value)
    {
        if(value.empty())
        {
            throw Exception(1,"[VectorX]: input vector should not be empty\n", __FILE__, __LINE__, __FUNCTION__);
        }
        _n = (int)value.size();
        _value = new T[_n]();
        memcpy(_value, value.data(), _n*sizeof(T));
    }

    VectorX(const VectorX& vec)
    {
        this->_n = vec._n;
        _value = new T(vec._n);
        if(vec._value!=nullptr)
        {
            memcpy(this->_value,vec._value,sizeof(T)*vec._n);
        }
    }

    VectorX(VectorX&& vec)
    {
        this->_n     = vec._n;
        this->_value = vec._value;
        vec.setDataNull();
    }

    VectorX &operator= (const VectorX& vec)
    {
        if(this!=&vec)
        {
            release();
            this->_n = vec._n;
            memcpy(this->_value,vec._value,sizeof(T)*vec._n);
        }

        return *this;
    }

    VectorX &operator= (VectorX&& vec)
    {
        if(this!=&vec)
        {
            release();
            this->_n     = vec._n;
            this->_value = vec._value;
            vec.setDataNull();
        }
        return *this;
    }

    VectorX &operator= (const std::vector<T> &vec)
    {
        release();
        this->_n = vec.size();
        memcpy(this->_value,vec.data(),sizeof(T)*_n);
        return *this;
    }

    void release()
    {
        if(this->_value!=nullptr)
        {
            delete[] this->_value;
            this->_value = nullptr;
        }
        this->_n = 0;
    }

    void setDataNull()
    {
        this->_n = 0;
        this->_value = nullptr;
    }

    inline void fill(const T &value)
    {
        for (int i = 0; i < _n; ++i)
        {
            this->_value[i] = value;
        }
    }

    inline void print()
    {
        std::cout<<"{ VectorX: "<<_n<<std::endl;
        if(isF32Vec())
        {
            for (int i = 0; i < _n; ++i)
            {
                std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<_value[i]<<" ";
            }
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < _n; ++i)
            {
                std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<_value[i]<<" ";
            }
        }
        else
        {
            for (int i = 0; i < _n; ++i)
            {
                std::cout<<_value[i]<<" ";
            }
        }

        std::cout<<";\n}"<<std::endl;
    }

    inline std::string toString() const
    {

        std::stringstream buf;

        buf<<"{ VectorX: "<<_n<<std::endl;
        if(isF32Vec())
        {
            for (int i = 0; i < _n; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<_value[i]<<" ";
            }
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < _n; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<_value[i]<<" ";
            }
        }
        else
        {
            for (int i = 0; i < _n; ++i)
            {
                buf<<_value[i]<<" ";
            }
        }

        buf<<";\n}"<<std::endl;

        return buf.str();
    }

    inline std::string toHtmlString() const
    {

        std::stringstream buf;

        buf<<"{ VectorX: "<<_n<<"<br/>";
        if(isF32Vec())
        {
            for (int i = 0; i < _n; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<_value[i]<<" ";
            }
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < _n; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<_value[i]<<" ";
            }
        }
        else
        {
            for (int i = 0; i < _n; ++i)
            {
                buf<<_value[i]<<" ";
            }
        }

        buf<<";\n}"<<"<br/>";

        return buf.str();
    }

    void setVal(const std::vector<T> &val)
    {
        if(val.size()!=_n)
        {
            throw Exception(1,"[VectorX]: set val num must equal data num! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        memcpy(this->_value, val.data(), sizeof(T)*_n);
    }

    void setVal(const int &index, const T &val)
    {
        if(index>(_n-1))
        {
            throw Exception(1,"[VectorX]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        this->_value[index] = val;
    }

    inline void zero()
    {
        for (int i = 0; i < _n; ++i)
        {
            this->_value[i] = 0;
        }
    }

    inline void reverseSign()
    {
        for (int i = 0; i < _n; ++i)
        {
            this->_value[i] = 0 - this->_value[i];
        }
    }

    bool isFuzzyNull() const
    {
        if(isF32Vec()) 

        {
            for (int i = 0; i < _n; ++i)
            {
                if(fabsf(this->_value[i])>MSNH_F32_EPS)
                {
                    return false;
                }
            }
            return true;
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < _n; ++i)
            {
                if(fabs(this->_value[i])>MSNH_F64_EPS)
                {
                    return false;
                }
            }
            return true;
        }
        else
        {
            for (int i = 0; i < _n; ++i)
            {
                if(this->_value[i]>0)
                {
                    return false;
                }
            }
            return true;
        }
    }

    inline bool isNan() const
    {
        for (int i = 0; i < _n; ++i)
        {
            if(std::isnan(static_cast<double>(this->_value[i])))
            {
                return true;
            }
        }
        return false;
    }

    inline bool isF32Vec() const
    {
        return std::is_same<T,float>::value;
    }

    inline bool isF64Vec() const
    {
        return std::is_same<T,double>::value;
    }

    VectorX normalized() const
    {
        if(!(isF32Vec() || isF64Vec()))
        {
            throw Exception(1, "[VectorX] normalize only f32 and f64 is supported!", __FILE__, __LINE__,__FUNCTION__);
        }

        T len = 0;

        VectorX vec(_n);

        for (int i = 0; i < _n; ++i)
        {
            len += this->_value[i]*this->_value[i];
        }

        if(isF32Vec())
        {
            if(fabsf(len - 1.0f) < MSNH_F32_EPS)
            {
                return *this;
            }

            if(fabsf(len) < MSNH_F32_EPS)
            {
                return vec;
            }

            len = sqrtf(len);
        }
        else if(isF64Vec())
        {
            if(fabs(len - 1.0) < MSNH_F64_EPS)
            {
                return *this;
            }

            if(fabs(len) < MSNH_F64_EPS)
            {
                return vec;
            }

            len = sqrt(len);
        }

        for (int i = 0; i < _n; ++i)
        {
            vec[i] = this->_value[i] / len;
        }

        return vec;
    }

    void normalize()
    {
        if(!(isF32Vec() || isF64Vec()))
        {
            throw Exception(1, "[VectorX] normalize only f32 and f64 is supported!", __FILE__, __LINE__,__FUNCTION__);
        }

        T len = 0;

        for (int i = 0; i < _n; ++i)
        {
            len += this->_value[i]*this->_value[i];
        }

        if(this->isF32Vec())
        {
            if(fabsf(len - 1.0f) < MSNH_F32_EPS || fabsf(len) < MSNH_F32_EPS)
            {
                return;
            }

            len = sqrtf(len);
        }
        else
        {
            if(fabs(len - 1.0) < MSNH_F64_EPS || fabs(len) < MSNH_F64_EPS)
            {
                return;
            }
            len = sqrt(len);
        }

        for (int i = 0; i < _n; ++i)
        {
            this->_value[i] = this->_value[i] / len;
        }
    }

    inline double length() const
    {
        double len = 0;
        for (int i = 0; i < _n; ++i)
        {
            len += this->_value[i]*this->_value[i];
        }
        return  sqrt(len);
    }

    inline double lengthSquared() const
    {
        double len = 0;
        for (int i = 0; i < _n; ++i)
        {
            len += this->_value[i]*this->_value[i];
        }
        return  len;
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
    inline double distanceToPoint(const VectorX &point) const
    {
        return (*this - point).length();
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
    inline double distanceToLine(const VectorX &point, const VectorX &direction) const
    {
        if(point.getN() != direction.getN())
        {
            throw Exception(1,"[VectorX]: data num of A and B not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        if(direction.getN()<2)
        {
            throw Exception(1,"[VectorX] only 2 dims+ is supported!",__FILE__,__LINE__,__FUNCTION__);
        }

        if(direction.isFuzzyNull())
        {
            return (*this - point).length();
        }

        VectorX p = point + VectorX::dotProduct((*this-point)*direction,direction);
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
    inline double distanceToPlane(const VectorX& plane, const VectorX& normal) const
    {
        if(plane.getN() != normal.getN())
        {
            throw Exception(1,"[VectorX]: data num of A and B not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        if(plane.getN()<3)
        {
            throw Exception(1,"[VectorX] only 3 dims+ is supported!",__FILE__,__LINE__,__FUNCTION__);
        }

        return dotProduct((*this-plane),normal);
    }

    inline static VectorX crossProduct(const VectorX &v1, const VectorX &v2)
    {
        if(v1.getN() != v2.getN())
        {
            throw Exception(1,"[VectorX]: data num of A and B not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        if(v1.getN()!=3)
        {
            throw Exception(1,"[VectorX] only 3 dims is supported!",__FILE__,__LINE__,__FUNCTION__);
        }

        return VectorX({ v1[1]*v2[2] - v1[2]*v2[1],
                         v1[2]*v2[0] - v1[0]*v2[2],
                         v1[0]*v2[1] - v1[1]*v2[0]});
    }

    inline static VectorX normal(const VectorX &v1, const VectorX &v2)
    {
        return crossProduct(v1,v2).normalized();
    }

    inline static VectorX normal(const VectorX &v1, const VectorX &v2, const VectorX &v3)
    {
        return crossProduct((v2-v1),(v3-v1)).normalized();
    }

    inline static T dotProduct(const VectorX &A, const VectorX &B)
    {
        if(A.getN() != B.getN())
        {
            throw Exception(1,"[VectorX]: data num of A and B not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        T finalVal = 0;
        for (int i = 0; i < A.getN(); ++i)
        {
            finalVal += A[i]*B[i];
        }
        return finalVal;
    }

    inline T operator [](const int &index) const
    {
        if(index > (_n-1))
        {
            throw Exception(1,"[VectorX]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        return _value[index];
    }

    inline T &operator [](const int &index)
    {
        if(index > (_n-1))
        {
            throw Exception(1,"[VectorX]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        return _value[index];
    }

    inline friend VectorX operator+ (const VectorX &A, const VectorX &B)
    {

        if(A.getN() != B.getN())
        {
            throw Exception(1,"[VectorX]: data num of A and B not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        VectorX tmp(A.getN());
        for (int i = 0; i < A.getN(); ++i)
        {
            tmp[i] = A[i] + B[i];
        }
        return tmp;
    }

    inline friend VectorX operator+ (T A, const VectorX &B)
    {
        VectorX tmp(B.getN());
        for (int i = 0; i < B.getN(); ++i)
        {
            tmp[i] = A + B[i];
        }
        return tmp;
    }

    inline friend VectorX operator+ (const VectorX &A, T B)
    {
        VectorX tmp(A.getN());
        for (int i = 0; i < A.getN(); ++i)
        {
            tmp[i] = A[i] + B;
        }
        return tmp;
    }

    inline friend VectorX operator- (const VectorX &A, const VectorX &B)
    {
        if(A.getN() != B.getN())
        {
            throw Exception(1,"[VectorX]: data num of A and B not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        VectorX tmp(A.getN());
        for (int i = 0; i < A.getN(); ++i)
        {
            tmp[i] = A[i] - B[i];
        }
        return tmp;
    }

    inline friend VectorX operator- (T A, const VectorX &B)
    {
        VectorX tmp(B.getN());
        for (int i = 0; i < B.getN(); ++i)
        {
            tmp[i] = A - B[i];
        }
        return tmp;
    }

    inline friend VectorX operator- (const VectorX &A, T B)
    {
        VectorX tmp(A.getN());
        for (int i = 0; i < A.getN(); ++i)
        {
            tmp[i] = A[i] - B;
        }
        return tmp;
    }

    inline friend VectorX operator- (const VectorX &A)
    {
        VectorX tmp(A.getN());
        for (int i = 0; i < A.getN(); ++i)
        {
            tmp[i] = 0 - A[i];
        }
        return tmp;
    }

    inline friend VectorX operator* (const VectorX &A, const VectorX &B)
    {
        if(A.getN() != B.getN())
        {
            throw Exception(1,"[VectorX]: data num of A and B not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        VectorX tmp(A.getN());
        for (int i = 0; i < A.getN(); ++i)
        {
            tmp[i] = A[i] * B[i];
        }
        return tmp;
    }

    inline friend VectorX operator* (T A, const VectorX &B)
    {
        VectorX tmp(B.getN());
        for (int i = 0; i < B.getN(); ++i)
        {
            tmp[i] = A * B[i];
        }
        return tmp;
    }

    inline friend VectorX operator* (const VectorX &A, T B)
    {
        VectorX tmp(A.getN());
        for (int i = 0; i < A.getN(); ++i)
        {
            tmp[i] = A[i] * B;
        }
        return tmp;
    }

    inline friend VectorX operator/ (const VectorX &A, T B)
    {
        VectorX tmp(A.getN());
        for (int i = 0; i < A.getN(); ++i)
        {
            tmp[i] = A[i] / B;
        }
        return tmp;
    }

    inline friend VectorX operator/ (const VectorX &A, const VectorX &B)
    {
        if(A.getN() != B.getN())
        {
            throw Exception(1,"[VectorX]: data num of A and B not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        VectorX tmp(A.getN());
        for (int i = 0; i < A.getN(); ++i)
        {
            tmp[i] = A[i] / B[i];
        }
        return tmp;
    }

    inline friend bool operator== (const VectorX &A, const VectorX &B)
    {
        if(A.getN() != B.getN())
        {
            return false;
        }

        if(A.isF32Vec())
        {
            for (int i = 0; i < A.getN(); ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F32_EPS)
                {
                    return false;
                }
            }
        }
        else if(A.isF64Vec())
        {
            for (int i = 0; i < A.getN(); ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F64_EPS)
                {
                    return false;
                }
            }
        }
        else
        {
            for (int i = 0; i < A.getN(); ++i)
            {
                if(A[i] != B[i])
                {
                    return false;
                }
            }

        }
        return true;
    }

    inline friend bool operator!= (const VectorX &A, const VectorX &B)
    {
        if(A.getN() != B.getN())
        {
            return true;
        }

        if(std::is_same<T,float>::value)
        {
            for (int i = 0; i < A.getN(); ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F32_EPS)
                {
                    return true;
                }
            }
        }
        else if(std::is_same<T,double>::value)
        {
            for (int i = 0; i < A.getN(); ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F64_EPS)
                {
                    return true;
                }
            }
        }
        else
        {
            for (int i = 0; i < A.getN(); ++i)
            {
                if(A[i] != B[i])
                {
                    return true;
                }
            }

        }
        return false;
    }

    inline VectorX &operator +=(const VectorX &A)
    {
        for (int i = 0; i < _n; ++i)
        {
            this->_value[i]+=A[i];
        }
        return *this;
    }

    inline VectorX &operator +=(T A)
    {
        for (int i = 0; i < _n; ++i)
        {
            this->_value[i]+=A;
        }
        return *this;
    }

    inline VectorX &operator -=(const VectorX &A)
    {
        for (int i = 0; i < _n; ++i)
        {
            this->_value[i]-=A[i];
        }
        return *this;
    }

    inline VectorX &operator -=(T A)
    {
        for (int i = 0; i < _n; ++i)
        {
            this->_value[i]-=A;
        }
        return *this;
    }

    inline VectorX &operator *=(const VectorX &A)
    {
        for (int i = 0; i < _n; ++i)
        {
            this->_value[i]*=A[i];
        }
        return *this;
    }

    inline VectorX &operator *=(T A)
    {
        for (int i = 0; i < _n; ++i)
        {
            this->_value[i]*=A;
        }
        return *this;
    }

    inline VectorX &operator /=(T A)
    {
        for (int i = 0; i < _n; ++i)
        {
            this->_value[i]/=A;
        }
        return *this;
    }

    inline T* getValue() const
    {
        return this->_value;
    }

    inline int getN() const
    {
        return this->_n;
    }

protected:
    int _n    = 0;
    T* _value = nullptr;
};

template<int N,typename T>
class Vector:public VectorX<T>
{
public:
    Vector():VectorX<T>(N){}
    Vector(const std::vector<T> &value):VectorX<T>(N)
    {
        this->setVal(value);
    }

    Vector(const Vector& vec):VectorX<T>(N)
    {
        memcpy(this->_value,vec._value,sizeof(T)*N);
    }

    Vector(Vector&& vec)
    {
        this->_n = N;
        this->_value = vec._value;
        vec.setDataNull();
    }

    Vector(const VectorX<T>& vec)
    {
        if(N!=vec._n)
        {
            throw Exception(1,"[Vector]: data num not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        this->_n = N;
        this->_value = new T[N]();
        memcpy(this->_value,vec._value,sizeof(T)*N);
    }

    Vector(VectorX<T>&& vec)
    {
        if(N!=vec.getN())
        {
            throw Exception(1,"[Vector]: data num not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        this->_n = N;
        this->_value = vec.getValue();
        vec.setDataNull();
    }

    Vector& operator =(const Vector &vec)
    {
        if(this!=&vec)
        {
            this->_n = N;
            this->_value = new T[N]();

            memcpy(this->_value,vec._value,sizeof(T)*N);
        }
        return *this;
    }

    Vector& operator =(Vector &&vec)
    {
        if(this!=&vec)
        {
            this->_n = N;
            this->_value = vec._value;
            vec.setDataNull();
        }
        return *this;
    }

    Vector& operator =(const VectorX<T> &vec)
    {
        if(N!=vec._n)
        {
            throw Exception(1,"[Vector]: data num not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        this->_n = N;
        this->_value = new T[N]();

        memcpy(this->_value,vec._value,sizeof(T)*N);
        return *this;
    }

    Vector& operator =(VectorX<T> &&vec)
    {
        if(N!=vec.getN())
        {
            throw Exception(1,"[Vector]: data num not equal! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        this->_n = N;
        this->_value = vec.getValue();
        vec.setDataNull();
        return *this;
    }
};

typedef VectorX<double> VectorXD;
typedef VectorX<float> VectorXF;

typedef Vector<3,double> EulerD;
typedef Vector<3,double> TranslationD;
typedef Vector<3,double> RotationVecD;
typedef Vector<3,double> LinearVelD;
typedef Vector<3,double> AngularVelD;
typedef Vector<2,double> Vector2D;
typedef Vector<3,double> Vector3D;
typedef Vector<5,double> Vector5D;
typedef Vector<4,double> Vector4D;
typedef Vector<6,double> Vector6D;
typedef Vector<7,double> Vector7D;

typedef Vector<3,float> EulerF;
typedef Vector<3,float> TranslationF;
typedef Vector<3,float> RotationVecF;
typedef Vector<3,float> LinearVelF;
typedef Vector<3,float> AngularVelF;
typedef Vector<2,float> Vector2F;
typedef Vector<3,float> Vector3F;
typedef Vector<4,float> Vector4F;
typedef Vector<5,float> Vector5F;
typedef Vector<6,float> Vector6F;
typedef Vector<7,float> Vector7F;
}
#endif 

