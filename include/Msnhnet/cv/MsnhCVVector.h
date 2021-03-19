#ifndef MSNHCVVECTOR_H
#define MSNHCVVECTOR_H

#include <Msnhnet/config/MsnhnetCfg.h>
#include <iostream>
#include <sstream>
namespace Msnhnet
{
template<int N,typename T>
class Vector
{
public:
    Vector(){}

    Vector(const std::vector<T> &val)
    {
        if(val.size()!=N)
        {
            throw Exception(1,"[Vector]: set val num must equal data num! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        for (int i = 0; i < N; ++i)
        {
            this->_value[i] = val[i];
        }
    }

    Vector(const Vector& vec)
    {
        memcpy(this->_value,vec._value,sizeof(T)*N);
    }

    Vector &operator= (const Vector &vec)
    {
        memcpy(this->_value,vec._value,sizeof(T)*N);
        return *this;
    }

    Vector &operator= (const std::vector<T> &val)
    {
        memcpy(this->_value,val.data(),sizeof(T)*N);
        return *this;
    }

    inline void fill(const T &value)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i] = value;
        }
    }

    inline void print()
    {
        std::cout<<"{ Vector: "<<N<<std::endl;
        if(isF32Vec())
        {
            for (int i = 0; i < N; ++i)
            {
                std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<_value[i]<<" ";
            }
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < N; ++i)
            {
                std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<_value[i]<<" ";
            }
        }
        else
        {
            for (int i = 0; i < N; ++i)
            {
                std::cout<<_value[i]<<" ";
            }
        }

        std::cout<<";\n}"<<std::endl;
    }

    inline std::string toString() const
    {

        std::stringstream buf;

        buf<<"{ Vector: "<<N<<std::endl;
        if(isF32Vec())
        {
            for (int i = 0; i < N; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<_value[i]<<" ";
            }
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < N; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<_value[i]<<" ";
            }
        }
        else
        {
            for (int i = 0; i < N; ++i)
            {
                buf<<_value[i]<<" ";
            }
        }

        buf<<";\n}"<<std::endl;
    }

    inline std::string toHtmlString() const
    {

        std::stringstream buf;

        buf<<"{ Vector: "<<N<<"<br/>";
        if(isF32Vec())
        {
            for (int i = 0; i < N; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<_value[i]<<" ";
            }
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < N; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<_value[i]<<" ";
            }
        }
        else
        {
            for (int i = 0; i < N; ++i)
            {
                buf<<_value[i]<<" ";
            }
        }

        buf<<";\n}"<<"<br/>";
    }

    void setVal(const std::vector<T> &val)
    {
        if(val.size()!=N)
        {
            throw Exception(1,"[Vector]: set val num must equal data num! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        for (int i = 0; i < N; ++i)
        {
            this->_value[i] = val[i];
        }
    }

    void setVal(const int &index, const T &val)
    {
        if(index>(N-1))
        {
            throw Exception(1,"[Vector]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        this->_value[index] = val;
    }

    bool isFuzzyNull() const
    {
        if(isF32Vec()) 

        {
            for (int i = 0; i < N; ++i)
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
            for (int i = 0; i < N; ++i)
            {
                if(abs(this->_value[i])>MSNH_F64_EPS)
                {
                    return false;
                }
            }
            return true;
        }
        else
        {
            for (int i = 0; i < N; ++i)
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
        for (int i = 0; i < N; ++i)
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

    Vector normalized() const
    {
        if(!(isF32Vec() || isF64Vec()))
        {
            throw Exception(1, "[Vector] normalize only f32 and f64 is supported!", __FILE__, __LINE__,__FUNCTION__);
        }

        T len = 0;

        Vector vec;

        for (int i = 0; i < N; ++i)
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
        else if(isF32Vec())
        {
            if(abs(len - 1.0) < MSNH_F64_EPS)
            {
                return *this;
            }

            if(abs(len) < MSNH_F64_EPS)
            {
                return vec;
            }

            len = sqrt(len);
        }

        for (int i = 0; i < N; ++i)
        {
            vec[i] = this->_value[i] / len;
        }

        return vec;
    }

    void normalize()
    {
        if(!(isF32Vec() || isF64Vec()))
        {
            throw Exception(1, "[Vector] normalize only f32 and f64 is supported!", __FILE__, __LINE__,__FUNCTION__);
        }

        T len = 0;

        for (int i = 0; i < N; ++i)
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
            if(abs(len - 1.0) < MSNH_F64_EPS || abs(len) < MSNH_F64_EPS)
            {
                return;
            }
            len = sqrt(len);
        }

        for (int i = 0; i < N; ++i)
        {
            this->_value[i] = this->_value[i] / len;
        }
    }

    inline double length() const
    {
        double len = 0;
        for (int i = 0; i < N; ++i)
        {
            len += this->_value[i]*this->_value[i];
        }
        return  sqrt(len);
    }

    inline double lengthSquared() const
    {
        double len = 0;
        for (int i = 0; i < N; ++i)
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
    inline double distanceToPoint(const Vector &point) const
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
    inline double distanceToLine(const Vector &point, const Vector &direction) const
    {
        if(N<2)
        {
            throw Exception(1,"[Vector] only 2 dims+ is supported!",__FILE__,__LINE__,__FUNCTION__);
        }

        if(direction.isFuzzyNull())
        {
            return (*this - point).length();
        }

        Vector p = point + Vector::dotProduct((*this-point)*direction,direction);
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
        if(N<3)
        {
           throw Exception(1,"[Vector] only 3 dims+ is supported!",__FILE__,__LINE__,__FUNCTION__);
        }

        return (*this-plane)*normal;
    }

    inline static Vector crossProduct(const Vector &v1, const Vector &v2)
    {
        if(N!=3)
        {
           throw Exception(1,"[Vector] only 3 dims is supported!",__FILE__,__LINE__,__FUNCTION__);
        }

        return Vector({ v1[1]*v2[2] - v1[2]*v2[1],
                        v1[2]*v2[0] - v1[0]*v2[2],
                        v1[0]*v2[1] - v1[1]*v2[0]});
    }

    inline static Vector normal(const Vector &v1, const Vector &v2)
    {
        return crossProduct(v1,v2).normalized();
    }

    inline static Vector normal(const Vector &v1, const Vector &v2, const Vector &v3)
    {
        return crossProduct((v2-v1),(v3-v1)).normalized();
    }

    inline static T dotProduct(const Vector &A, const Vector &B)
    {
        T finalVal = 0;
        for (int i = 0; i < N; ++i)
        {
            finalVal += A[i]*B[i];
        }
        return finalVal;
    }

    inline T operator [](const int &index) const
    {
        if(index > (N-1))
        {
            throw Exception(1,"[Vector]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        return _value[index];
    }

    inline T &operator [](const int &index)
    {
        if(index > (N-1))
        {
            throw Exception(1,"[Vector]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        return _value[index];
    }

    inline friend Vector operator+ (const Vector &A, const Vector &B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] + B[i];
        }
        return tmp;
    }

    inline friend Vector operator+ (T A, const Vector &B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A + B[i];
        }
        return tmp;
    }

    inline friend Vector operator+ (const Vector &A, T B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] + B;
        }
        return tmp;
    }

    inline friend Vector operator- (const Vector &A, const Vector &B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] - B[i];
        }
        return tmp;
    }

    inline friend Vector operator- (T A, const Vector &B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A - B[i];
        }
        return tmp;
    }

    inline friend Vector operator- (const Vector &A, T B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] - B;
        }
        return tmp;
    }

    inline friend Vector operator- (const Vector &A)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = 0 - A[i];
        }
        return tmp;
    }

    inline friend Vector operator* (const Vector &A, const Vector &B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] * B[i];
        }
        return tmp;
    }

    inline friend Vector operator* (T A, const Vector &B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A * B[i];
        }
        return tmp;
    }

    inline friend Vector operator* (const Vector &A, T B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] * B;
        }
        return tmp;
    }

    inline friend Vector operator/ (const Vector &A, T B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] / B;
        }
        return tmp;
    }

    inline friend Vector operator/ (const Vector &A, const Vector &B)
    {
        Vector tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] / B[i];
        }
        return tmp;
    }

    inline friend bool operator== (const Vector &A, const Vector &B)
    {
        if(A.isF32Vec())
        {
            for (int i = 0; i < N; ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F32_EPS)
                {
                    return false;
                }
            }
        }
        else if(A.isF64Vec())
        {
            for (int i = 0; i < N; ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F64_EPS)
                {
                    return false;
                }
            }
        }
        else
        {
            for (int i = 0; i < N; ++i)
            {
                if(A[i] != B[i])
                {
                    return false;
                }
            }

        }
        return true;
    }

    inline friend bool operator!= (const Vector &A, const Vector &B)
    {
        if(std::is_same<T,float>::value)
        {
            for (int i = 0; i < N; ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F32_EPS)
                {
                    return true;
                }
            }
        }
        else if(std::is_same<T,double>::value)
        {
            for (int i = 0; i < N; ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F64_EPS)
                {
                    return true;
                }
            }
        }
        else
        {
            for (int i = 0; i < N; ++i)
            {
                if(A[i] != B[i])
                {
                    return true;
                }
            }

        }
        return false;
    }

    inline Vector &operator +=(const Vector &A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]+=A[i];
        }
        return *this;
    }

    inline Vector &operator +=(T A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]+=A;
        }
        return *this;
    }

    inline Vector &operator -=(const Vector &A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]-=A[i];
        }
        return *this;
    }

    inline Vector &operator -=(T A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]-=A;
        }
        return *this;
    }

    inline Vector &operator *=(const Vector &A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]*=A[i];
        }
        return *this;
    }

    inline Vector &operator *=(T A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]*=A;
        }
        return *this;
    }

    inline Vector &operator /=(T A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]/=A;
        }
        return *this;
    }

private:
    T _value[N] = {0};
};

typedef Vector<3,double> EulerD;
typedef Vector<3,double> TranslationD;
typedef Vector<3,double> RotationVecD;
typedef Vector<2,double> Vector2D;
typedef Vector<3,double> Vector3D;
typedef Vector<4,double> Vector4D;

typedef Vector<3,float> EulerF;
typedef Vector<3,float> TranslationF;
typedef Vector<3,float> RotationVecF;
typedef Vector<2,float> Vector2F;
typedef Vector<3,float> Vector3F;
typedef Vector<4,float> Vector4F;

}
#endif 

