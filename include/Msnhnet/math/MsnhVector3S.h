#ifndef VECTOR3S_H
#define VECTOR3S_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace Msnhnet
{
class MsnhNet_API Vector3DS
{
public :
    double val[3];

    inline Vector3DS(){val[0]=val[1]=val[2]=0;}

    inline Vector3DS(const double &x,const double &y,const double &z)
    {
        val[0] = x;
        val[1] = y;
        val[2] = z;
    }

    inline Vector3DS(const std::vector<double>& vec)
    {
        assert(vec.size()==3);

        val[0] = vec[0];
        val[1] = vec[1];
        val[2] = vec[2];
    }

    inline Vector3DS(const Vector3DS& vec)
    {
        val[0] = vec.val[0];
        val[1] = vec.val[1];
        val[2] = vec.val[2];
    }

    inline Vector3DS& operator =(const Vector3DS& vec)
    {
        if(this!=&vec)
        {
            val[0] = vec.val[0];
            val[1] = vec.val[1];
            val[2] = vec.val[2];
        }
        return *this;
    }

    inline void setval(const double &x,const double &y,const double &z)
    {
        val[0] = x;
        val[1] = y;
        val[2] = z;
    }

    inline double operator [](const uint8_t &index) const
    {
        assert(index < 3);
        return val[index];
    }

    inline double &operator [](const uint8_t &index)
    {
        assert(index < 3);
        return val[index];
    }

    void print();

    std::string toString() const;

    std::string toHtmlString() const;

    inline friend bool operator ==(const Vector3DS& A, const Vector3DS& B)
    {
        if(abs(A.val[0]-B.val[0]) < MSNH_F64_EPS &&
                abs(A.val[1]-B.val[1]) < MSNH_F64_EPS &&
                abs(A.val[2]-B.val[2]) < MSNH_F64_EPS)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    inline friend bool operator !=(const Vector3DS& A, const Vector3DS& B)
    {
        if(abs(A.val[0]-B.val[0]) < MSNH_F64_EPS &&
                abs(A.val[1]-B.val[1]) < MSNH_F64_EPS &&
                abs(A.val[2]-B.val[2]) < MSNH_F64_EPS)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    inline bool isFuzzyNull() const
    {
        for (int i = 0; i < 3; ++i)
        {
            if(abs(val[i])>MSNH_F64_EPS)
            {
                return false;
            }
        }
        return true;
    }

    inline bool isNan() const
    {
        for (int i = 0; i < 3; ++i)
        {
            if(std::isnan(static_cast<double>(val[i])))
            {
                return true;
            }
        }
        return false;
    }

    inline bool closeToEps(const double &eps)
    {
        for (int i = 0; i < 3; ++i)
        {
            if(abs(val[i]-eps)>MSNH_F64_EPS)
            {
                return false;
            }
        }
        return true;
    }

    inline friend Vector3DS operator +(const Vector3DS& A, const Vector3DS& B)
    {
        Vector3DS tmp;
        tmp.val[0] = A.val[0]+B.val[0];
        tmp.val[1] = A.val[1]+B.val[1];
        tmp.val[2] = A.val[2]+B.val[2];
        return tmp;
    }

    inline friend Vector3DS operator +(const Vector3DS& A, const double& b)
    {
        Vector3DS tmp;
        tmp.val[0] = A.val[0] + b;
        tmp.val[1] = A.val[1] + b;
        tmp.val[2] = A.val[2] + b;
        return tmp;
    }

    inline friend Vector3DS operator +(const double& a, const Vector3DS& B)
    {
        Vector3DS tmp;
        tmp.val[0] = B.val[0] + a;
        tmp.val[1] = B.val[1] + a;
        tmp.val[2] = B.val[2] + a;
        return tmp;
    }

    inline Vector3DS &operator +=(const Vector3DS& A)
    {
        val[0] =  val[0] + A.val[0] ;
        val[1] =  val[1] + A.val[1] ;
        val[2] =  val[2] + A.val[2] ;
        return *this;
    }

    inline Vector3DS &operator +=(const double& a)
    {
        val[0] =  val[0] + a ;
        val[1] =  val[1] + a ;
        val[2] =  val[2] + a ;
        return *this;
    }

    inline friend Vector3DS operator -(const Vector3DS& A, const Vector3DS& B)
    {
        Vector3DS tmp;
        tmp.val[0] = A.val[0]-B.val[0];
        tmp.val[1] = A.val[1]-B.val[1];
        tmp.val[2] = A.val[2]-B.val[2];
        return tmp;
    }

    inline friend Vector3DS operator -(const Vector3DS& A, const double& b)
    {
        Vector3DS tmp;
        tmp.val[0] = A.val[0] - b;
        tmp.val[1] = A.val[1] - b;
        tmp.val[2] = A.val[2] - b;
        return tmp;
    }

    inline friend Vector3DS operator -(const double& a, const Vector3DS& B)
    {
        Vector3DS tmp;
        tmp.val[0] = a - B.val[0];
        tmp.val[1] = a - B.val[1];
        tmp.val[2] = a - B.val[2];
        return tmp;
    }

    inline Vector3DS &operator -=(const Vector3DS& A)
    {
        val[0] =  val[0] - A.val[0] ;
        val[1] =  val[1] - A.val[1] ;
        val[2] =  val[2] - A.val[2] ;
        return *this;
    }

    inline Vector3DS &operator -=(const double& a)
    {
        val[0] =  val[0] - a ;
        val[1] =  val[1] - a ;
        val[2] =  val[2] - a ;
        return *this;
    }

    inline friend Vector3DS operator *(const Vector3DS& A, const Vector3DS& B)
    {
        Vector3DS tmp;
        tmp.val[0] = A.val[0]*B.val[0];
        tmp.val[1] = A.val[1]*B.val[1];
        tmp.val[2] = A.val[2]*B.val[2];
        return tmp;
    }

    inline friend Vector3DS operator *(const Vector3DS& A, const double& b)
    {
        Vector3DS tmp;
        tmp.val[0] = A.val[0] * b;
        tmp.val[1] = A.val[1] * b;
        tmp.val[2] = A.val[2] * b;
        return tmp;
    }

    inline friend Vector3DS operator *(const double& a, const Vector3DS& B)
    {
        Vector3DS tmp;
        tmp.val[0] = a * B.val[0];
        tmp.val[1] = a * B.val[1];
        tmp.val[2] = a * B.val[2];
        return tmp;
    }

    inline Vector3DS &operator *=(const Vector3DS& A)
    {
        val[0] =  val[0] * A.val[0] ;
        val[1] =  val[1] * A.val[1] ;
        val[2] =  val[2] * A.val[2] ;
        return *this;
    }

    inline Vector3DS &operator *=(const double& a)
    {
        val[0] =  val[0] * a ;
        val[1] =  val[1] * a ;
        val[2] =  val[2] * a ;
        return *this;
    }

    inline static Vector3DS crossProduct(const Vector3DS& A, const Vector3DS& B)
    {
        Vector3DS tmp;
        tmp.val[0] = A.val[1]*B.val[2]-A.val[2]*B.val[1];
        tmp.val[1] = A.val[2]*B.val[0]-A.val[0]*B.val[2];
        tmp.val[2] = A.val[0]*B.val[1]-A.val[1]*B.val[0];
        return tmp;
    }

    inline static double dotProduct(const Vector3DS& A, const Vector3DS& B)
    {
        return A.val[0]*B.val[0] + A.val[1]*B.val[1] + A.val[2]*B.val[2];
    }

    inline friend Vector3DS operator /(const Vector3DS& A, const double& b)
    {
        Vector3DS tmp;
        tmp.val[0] = A.val[0] / b;
        tmp.val[1] = A.val[1] / b;
        tmp.val[2] = A.val[2] / b;
        return tmp;
    }

    inline friend Vector3DS operator /(const Vector3DS& A, const Vector3DS& B)
    {
        Vector3DS tmp;
        tmp.val[0] = A.val[0] / B.val[0];
        tmp.val[1] = A.val[1] / B.val[1];
        tmp.val[2] = A.val[2] / B.val[2];
        return tmp;
    }

    inline Vector3DS &operator /=(const Vector3DS& A)
    {
        val[0] =  val[0] / A.val[0] ;
        val[1] =  val[1] / A.val[1] ;
        val[2] =  val[2] / A.val[2] ;
        return *this;
    }

    inline Vector3DS &operator /=(const double& a)
    {
        val[0] =  val[0] / a ;
        val[1] =  val[1] / a ;
        val[2] =  val[2] / a ;
        return *this;
    }

    inline Vector3DS normalized()
    {
        Vector3DS vec;

        double len = val[0]*val[0] + val[1]*val[1] + val[2]*val[2];

        if(abs(len - 1.0) < MSNH_F64_EPS)
        {
            return *this;
        }

        if(abs(len) < MSNH_F64_EPS)
        {
            return vec;
        }

        len = sqrt(len);

        vec.val[0] = val[0] / len;
        vec.val[1] = val[1] / len;
        vec.val[2] = val[2] / len;
        return vec;
    }

    inline void normalize()
    {
        double len = val[0]*val[0] + val[1]*val[1] + val[2]*val[2];

        if(abs(len - 1.0) < MSNH_F64_EPS || abs(len) < MSNH_F64_EPS)
        {
            return;
        }

        len = sqrt(len);

        val[0] = val[0] / len;
        val[1] = val[1] / len;
        val[2] = val[2] / len;
    }

    inline double length() const
    {
        return sqrt(val[0]*val[0] + val[1]*val[1] + val[2]*val[2]);
    }

    inline double lengthSquared() const
    {
        return val[0]*val[0] + val[1]*val[1] + val[2]*val[2];
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
    inline double distanceToPoint(const Vector3DS &point) const
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
    inline double distanceToLine(const Vector3DS &point, const Vector3DS &direction) const
    {
        if(direction.isFuzzyNull())
        {
            return (*this - point).length();
        }

        Vector3DS p = point + Vector3DS::dotProduct((*this-point)*direction,direction);
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
    inline double distanceToPlane(const Vector3DS& plane, const Vector3DS& normal) const
    {
        return dotProduct((*this-plane),normal);
    }

    inline static Vector3DS normal(const Vector3DS &v1, const Vector3DS &v2)
    {
        return crossProduct(v1,v2).normalized();
    }

    inline static Vector3DS normal(const Vector3DS &v1, const Vector3DS &v2, const Vector3DS &v3)
    {
        return crossProduct((v2-v1),(v3-v1)).normalized();
    }

};

class Vector3FS
{
public :
    float val[3];

    inline Vector3FS(){val[0]=val[1]=val[2]=0;}

    inline Vector3FS(const float &x,const float &y,const float &z)
    {
        val[0] = x;
        val[1] = y;
        val[2] = z;
    }

    inline Vector3FS(const std::vector<float>& vec)
    {
        assert(vec.size()==3);

        val[0] = vec[0];
        val[1] = vec[1];
        val[2] = vec[2];
    }

    inline Vector3FS(const Vector3FS& vec)
    {
        val[0] = vec.val[0];
        val[1] = vec.val[1];
        val[2] = vec.val[2];
    }

    inline Vector3FS& operator =(const Vector3FS& vec)
    {
        if(this!=&vec)
        {
            val[0] = vec.val[0];
            val[1] = vec.val[1];
            val[2] = vec.val[2];
        }
        return *this;
    }

    inline void setval(const float &x,const float &y,const float &z)
    {
        val[0] = x;
        val[1] = y;
        val[2] = z;
    }

    inline float operator [](const uint8_t &index) const
    {
        assert(index < 3);
        return val[index];
    }

    inline float &operator [](const uint8_t &index)
    {
        assert(index < 3);
        return val[index];
    }

    void print();

    std::string toString() const;

    std::string toHtmlString() const;

    inline friend bool operator ==(const Vector3FS& A, const Vector3FS& B)
    {
        if(fabsf(A.val[0]-B.val[0]) < MSNH_F32_EPS &&
                fabsf(A.val[1]-B.val[1]) < MSNH_F32_EPS &&
                fabsf(A.val[2]-B.val[2]) < MSNH_F32_EPS)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    inline friend bool operator !=(const Vector3FS& A, const Vector3FS& B)
    {
        if(fabsf(A.val[0]-B.val[0]) < MSNH_F32_EPS &&
                fabsf(A.val[1]-B.val[1]) < MSNH_F32_EPS &&
                fabsf(A.val[2]-B.val[2]) < MSNH_F32_EPS)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    inline bool isFuzzyNull() const
    {
        for (int i = 0; i < 3; ++i)
        {
            if(fabsf(val[i])>MSNH_F32_EPS)
            {
                return false;
            }
        }
        return true;
    }

    inline bool isNan() const
    {
        for (int i = 0; i < 3; ++i)
        {
            if(std::isnan(static_cast<float>(val[i])))
            {
                return true;
            }
        }
        return false;
    }

    inline bool closeToEps(const float &eps)
    {
        for (int i = 0; i < 3; ++i)
        {
            if(abs(val[i]-eps)>MSNH_F32_EPS)
            {
                return false;
            }
        }
        return true;
    }

    inline friend Vector3FS operator +(const Vector3FS& A, const Vector3FS& B)
    {
        Vector3FS tmp;
        tmp.val[0] = A.val[0]+B.val[0];
        tmp.val[1] = A.val[1]+B.val[1];
        tmp.val[2] = A.val[2]+B.val[2];
        return tmp;
    }

    inline friend Vector3FS operator +(const Vector3FS& A, const float& b)
    {
        Vector3FS tmp;
        tmp.val[0] = A.val[0] + b;
        tmp.val[1] = A.val[1] + b;
        tmp.val[2] = A.val[2] + b;
        return tmp;
    }

    inline friend Vector3FS operator +(const float& a, const Vector3FS& B)
    {
        Vector3FS tmp;
        tmp.val[0] = B.val[0] + a;
        tmp.val[1] = B.val[1] + a;
        tmp.val[2] = B.val[2] + a;
        return tmp;
    }

    inline Vector3FS &operator +=(const Vector3FS& A)
    {
        val[0] =  val[0] + A.val[0] ;
        val[1] =  val[1] + A.val[1] ;
        val[2] =  val[2] + A.val[2] ;
        return *this;
    }

    inline Vector3FS &operator +=(const float& a)
    {
        val[0] =  val[0] + a ;
        val[1] =  val[1] + a ;
        val[2] =  val[2] + a ;
        return *this;
    }

    inline friend Vector3FS operator -(const Vector3FS& A, const Vector3FS& B)
    {
        Vector3FS tmp;
        tmp.val[0] = A.val[0]-B.val[0];
        tmp.val[1] = A.val[1]-B.val[1];
        tmp.val[2] = A.val[2]-B.val[2];
        return tmp;
    }

    inline friend Vector3FS operator -(const Vector3FS& A, const float& b)
    {
        Vector3FS tmp;
        tmp.val[0] = A.val[0] - b;
        tmp.val[1] = A.val[1] - b;
        tmp.val[2] = A.val[2] - b;
        return tmp;
    }

    inline friend Vector3FS operator -(const float& a, const Vector3FS& B)
    {
        Vector3FS tmp;
        tmp.val[0] = a - B.val[0];
        tmp.val[1] = a - B.val[1];
        tmp.val[2] = a - B.val[2];
        return tmp;
    }

    inline Vector3FS &operator -=(const Vector3FS& A)
    {
        val[0] =  val[0] - A.val[0] ;
        val[1] =  val[1] - A.val[1] ;
        val[2] =  val[2] - A.val[2] ;
        return *this;
    }

    inline Vector3FS &operator -=(const float& a)
    {
        val[0] =  val[0] - a ;
        val[1] =  val[1] - a ;
        val[2] =  val[2] - a ;
        return *this;
    }

    inline friend Vector3FS operator *(const Vector3FS& A, const Vector3FS& B)
    {
        Vector3FS tmp;
        tmp.val[0] = A.val[0]*B.val[0];
        tmp.val[1] = A.val[1]*B.val[1];
        tmp.val[2] = A.val[2]*B.val[2];
        return tmp;
    }

    inline friend Vector3FS operator *(const Vector3FS& A, const float& b)
    {
        Vector3FS tmp;
        tmp.val[0] = A.val[0] * b;
        tmp.val[1] = A.val[1] * b;
        tmp.val[2] = A.val[2] * b;
        return tmp;
    }

    inline friend Vector3FS operator *(const float& a, const Vector3FS& B)
    {
        Vector3FS tmp;
        tmp.val[0] = a * B.val[0];
        tmp.val[1] = a * B.val[1];
        tmp.val[2] = a * B.val[2];
        return tmp;
    }

    inline Vector3FS &operator *=(const Vector3FS& A)
    {
        val[0] =  val[0] * A.val[0] ;
        val[1] =  val[1] * A.val[1] ;
        val[2] =  val[2] * A.val[2] ;
        return *this;
    }

    inline Vector3FS &operator *=(const float& a)
    {
        val[0] =  val[0] * a ;
        val[1] =  val[1] * a ;
        val[2] =  val[2] * a ;
        return *this;
    }

    inline static Vector3FS crossProduct(const Vector3FS& A, const Vector3FS& B)
    {
        Vector3FS tmp;
        tmp.val[0] = A.val[1]*B.val[2]-A.val[2]*B.val[1];
        tmp.val[1] = A.val[2]*B.val[0]-A.val[0]*B.val[2];
        tmp.val[2] = A.val[0]*B.val[1]-A.val[1]*B.val[0];
        return tmp;
    }

    inline static float dotProduct(const Vector3FS& A, const Vector3FS& B)
    {
        return A.val[0]*B.val[0] + A.val[1]*B.val[1] + A.val[2]*B.val[2];
    }

    inline friend Vector3FS operator /(const Vector3FS& A, const float& b)
    {
        Vector3FS tmp;
        tmp.val[0] = A.val[0] / b;
        tmp.val[1] = A.val[1] / b;
        tmp.val[2] = A.val[2] / b;
        return tmp;
    }

    inline friend Vector3FS operator /(const Vector3FS& A, const Vector3FS& B)
    {
        Vector3FS tmp;
        tmp.val[0] = A.val[0] / B.val[0];
        tmp.val[1] = A.val[1] / B.val[1];
        tmp.val[2] = A.val[2] / B.val[2];
        return tmp;
    }

    inline Vector3FS &operator /=(const Vector3FS& A)
    {
        val[0] =  val[0] / A.val[0] ;
        val[1] =  val[1] / A.val[1] ;
        val[2] =  val[2] / A.val[2] ;
        return *this;
    }

    inline Vector3FS &operator /=(const float& a)
    {
        val[0] =  val[0] / a ;
        val[1] =  val[1] / a ;
        val[2] =  val[2] / a ;
        return *this;
    }

    inline Vector3FS normalized()
    {
        Vector3FS vec;

        float len = val[0]*val[0] + val[1]*val[1] + val[2]*val[2];

        if(fabsf(len - 1.0) < MSNH_F32_EPS)
        {
            return *this;
        }

        if(fabsf(len) < MSNH_F32_EPS)
        {
            return vec;
        }

        len = sqrtf(len);

        vec.val[0] = val[0] / len;
        vec.val[1] = val[1] / len;
        vec.val[2] = val[2] / len;
        return vec;
    }

    inline void normalize()
    {
        float len = val[0]*val[0] + val[1]*val[1] + val[2]*val[2];

        if(fabsf(len - 1.0) < MSNH_F32_EPS || fabsf(len) < MSNH_F32_EPS)
        {
            return;
        }

        len = sqrtf(len);

        val[0] = val[0] / len;
        val[1] = val[1] / len;
        val[2] = val[2] / len;
    }

    inline float length() const
    {
        return sqrtf(val[0]*val[0] + val[1]*val[1] + val[2]*val[2]);
    }

    inline float lengthSquared() const
    {
        return val[0]*val[0] + val[1]*val[1] + val[2]*val[2];
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
    inline float distanceToPoint(const Vector3FS &point) const
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
    inline float distanceToLine(const Vector3FS &point, const Vector3FS &direction) const
    {
        if(direction.isFuzzyNull())
        {
            return (*this - point).length();
        }

        Vector3FS p = point + Vector3FS::dotProduct((*this-point)*direction,direction);
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
    inline float distanceToPlane(const Vector3FS& plane, const Vector3FS& normal) const
    {
        return dotProduct((*this-plane),normal);
    }

    inline static Vector3FS normal(const Vector3FS &v1, const Vector3FS &v2)
    {
        return crossProduct(v1,v2).normalized();
    }

    inline static Vector3FS normal(const Vector3FS &v1, const Vector3FS &v2, const Vector3FS &v3)
    {
        return crossProduct((v2-v1),(v3-v1)).normalized();
    }

};

typedef Vector3DS EulerDS;
typedef Vector3DS TranslationDS;
typedef Vector3DS RotationVecDS;
typedef Vector3DS LinearVelDS;
typedef Vector3DS AngularVelDS;

typedef Vector3FS EulerFS;
typedef Vector3FS TranslationFS;
typedef Vector3FS RotationVecFS;
typedef Vector3FS LinearVelFS;
typedef Vector3FS AngularVelFS;

}

#endif 

