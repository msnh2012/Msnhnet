#ifndef MSNHROTATIONMAT_H
#define MSNHROTATIONMAT_H

#include "Msnhnet/math/MsnhVector3S.h"
namespace Msnhnet
{
class MsnhNet_API RotationMatDS
{
public :
    double val[9];
    inline RotationMatDS()
    {
        val[0] = 1; val[1] = 0; val[2] = 0;
        val[3] = 0; val[4] = 1; val[5] = 0;
        val[6] = 0; val[7] = 0; val[8] = 1;
    }

    inline RotationMatDS(const std::vector<double>& vec)
    {
        assert(vec.size()==9);
        val[0] = vec[0]; val[1] = vec[1]; val[2] = vec[2];
        val[3] = vec[3]; val[4] = vec[4]; val[5] = vec[5];
        val[6] = vec[6]; val[7] = vec[7]; val[8] = vec[8];
    }

    inline RotationMatDS(const Vector3DS& x, const Vector3DS& y, const Vector3DS& z)
    {
        val[0] = x.val[0]; val[1] = y.val[0]; val[2] = z.val[0];
        val[3] = x.val[1]; val[4] = y.val[1]; val[5] = z.val[1];
        val[6] = x.val[2]; val[7] = y.val[2]; val[8] = z.val[2];
    }

    inline RotationMatDS(const RotationMatDS& vec)
    {
        val[0] = vec.val[0]; val[1] = vec.val[1]; val[2] = vec.val[2];
        val[3] = vec.val[3]; val[4] = vec.val[4]; val[5] = vec.val[5];
        val[6] = vec.val[6]; val[7] = vec.val[7]; val[8] = vec.val[8];
    }

    inline RotationMatDS& operator =(const RotationMatDS& vec)
    {
        if(this!=&vec)
        {
            val[0] = vec.val[0]; val[1] = vec.val[1]; val[2] = vec.val[2];
            val[3] = vec.val[3]; val[4] = vec.val[4]; val[5] = vec.val[5];
            val[6] = vec.val[6]; val[7] = vec.val[7]; val[8] = vec.val[8];
        }
        return *this;
    }

    inline double operator ()(const uint8_t& i, const uint8_t& j) const
    {
        int n = i*3 + j;
        assert(n < 9);
        return val[n];
    }

    inline double &operator ()(const uint8_t& i, const uint8_t& j)
    {
        int n = i*3 + j;
        assert(n < 9);
        return val[n];
    }

    inline void setVal(const std::vector<double>& vec)
    {
        assert(vec.size()==9);
        val[0] = vec[0]; val[1] = vec[1]; val[2] = vec[2];
        val[3] = vec[3]; val[4] = vec[4]; val[5] = vec[5];
        val[6] = vec[6]; val[7] = vec[7]; val[8] = vec[8];
    }

    inline void setX(const Vector3DS& x)
    {
        val[0] = x.val[0];
        val[3] = x.val[1];
        val[6] = x.val[2];
    }

    inline void setY(const Vector3DS& y)
    {
        val[1] = y.val[0];
        val[4] = y.val[1];
        val[7] = y.val[2];
    }

    inline void setZ(const Vector3DS& z)
    {
        val[2] = z.val[0];
        val[5] = z.val[1];
        val[8] = z.val[2];
    }

    inline Vector3DS getX() const
    {
        return Vector3DS(val[0],val[3],val[6]);
    }

    inline Vector3DS getY() const
    {
        return Vector3DS(val[1],val[4],val[7]);
    }

    inline Vector3DS getZ() const
    {
        return Vector3DS(val[2],val[5],val[8]);
    }

    double getRotAngle(Vector3DS &axis, double eps) const;

    inline Vector3DS getRot() const
    {
        Vector3DS axis;
        double angle = getRotAngle(axis, 10e-6);
        return axis*angle;
    }

    inline friend bool operator ==(const RotationMatDS& A, const RotationMatDS& B)
    {
        if(std::fabs(A.val[0]-B.val[0]) < MSNH_F64_EPS && std::fabs(A.val[1]-B.val[1]) < MSNH_F64_EPS && std::fabs(A.val[2]-B.val[2]) < MSNH_F64_EPS &&
                std::fabs(A.val[3]-B.val[3]) < MSNH_F64_EPS && std::fabs(A.val[4]-B.val[4]) < MSNH_F64_EPS && std::fabs(A.val[5]-B.val[5]) < MSNH_F64_EPS &&
                std::fabs(A.val[6]-B.val[6]) < MSNH_F64_EPS && std::fabs(A.val[7]-B.val[7]) < MSNH_F64_EPS && std::fabs(A.val[8]-B.val[8]) < MSNH_F64_EPS)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    inline friend bool operator !=(const RotationMatDS& A, const RotationMatDS& B)
    {
        if(std::fabs(A.val[0]-B.val[0]) < MSNH_F64_EPS && std::fabs(A.val[1]-B.val[1]) < MSNH_F64_EPS && std::fabs(A.val[2]-B.val[2]) < MSNH_F64_EPS &&
                std::fabs(A.val[3]-B.val[3]) < MSNH_F64_EPS && std::fabs(A.val[4]-B.val[4]) < MSNH_F64_EPS && std::fabs(A.val[5]-B.val[5]) < MSNH_F64_EPS &&
                std::fabs(A.val[6]-B.val[6]) < MSNH_F64_EPS && std::fabs(A.val[7]-B.val[7]) < MSNH_F64_EPS && std::fabs(A.val[8]-B.val[8]) < MSNH_F64_EPS)
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
        if(std::fabs(val[0]) < MSNH_F64_EPS && std::fabs(val[1]) < MSNH_F64_EPS && std::fabs(val[2]) < MSNH_F64_EPS &&
                std::fabs(val[3]) < MSNH_F64_EPS && std::fabs(val[4]) < MSNH_F64_EPS && std::fabs(val[5]) < MSNH_F64_EPS &&
                std::fabs(val[6]) < MSNH_F64_EPS && std::fabs(val[7]) < MSNH_F64_EPS && std::fabs(val[8]) < MSNH_F64_EPS)
        {
            return true;
        }
        return false;
    }

    inline bool closeToEps(const double &eps)
    {
        if(std::fabs(val[0]-eps) < MSNH_F64_EPS && std::fabs(val[1]-eps) < MSNH_F64_EPS && std::fabs(val[2]-eps) < MSNH_F64_EPS &&
                std::fabs(val[3]-eps) < MSNH_F64_EPS && std::fabs(val[4]-eps) < MSNH_F64_EPS && std::fabs(val[5]-eps) < MSNH_F64_EPS &&
                std::fabs(val[6]-eps) < MSNH_F64_EPS && std::fabs(val[7]-eps) < MSNH_F64_EPS && std::fabs(val[8]-eps) < MSNH_F64_EPS)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void print() const;

    std::string toString() const;

    std::string toHtmlString() const;

    inline friend RotationMatDS operator *(const RotationMatDS& A, const RotationMatDS& B)
    {
        RotationMatDS tmp;
        tmp.val[0] = A.val[0]*B.val[0]+A.val[1]*B.val[3]+A.val[2]*B.val[6];
        tmp.val[1] = A.val[0]*B.val[1]+A.val[1]*B.val[4]+A.val[2]*B.val[7];
        tmp.val[2] = A.val[0]*B.val[2]+A.val[1]*B.val[5]+A.val[2]*B.val[8];
        tmp.val[3] = A.val[3]*B.val[0]+A.val[4]*B.val[3]+A.val[5]*B.val[6];
        tmp.val[4] = A.val[3]*B.val[1]+A.val[4]*B.val[4]+A.val[5]*B.val[7];
        tmp.val[5] = A.val[3]*B.val[2]+A.val[4]*B.val[5]+A.val[5]*B.val[8];
        tmp.val[6] = A.val[6]*B.val[0]+A.val[7]*B.val[3]+A.val[8]*B.val[6];
        tmp.val[7] = A.val[6]*B.val[1]+A.val[7]*B.val[4]+A.val[8]*B.val[7];
        tmp.val[8] = A.val[6]*B.val[2]+A.val[7]*B.val[5]+A.val[8]*B.val[8];
        return tmp;
    }

    inline friend RotationMatDS operator *(const RotationMatDS& A, const double& b)
    {
        RotationMatDS tmp;
        tmp.val[0] = A.val[0]*b;
        tmp.val[1] = A.val[1]*b;
        tmp.val[2] = A.val[2]*b;
        tmp.val[3] = A.val[3]*b;
        tmp.val[4] = A.val[4]*b;
        tmp.val[5] = A.val[5]*b;
        tmp.val[6] = A.val[6]*b;
        tmp.val[7] = A.val[7]*b;
        tmp.val[8] = A.val[8]*b;
        return tmp;
    }

    inline friend RotationMatDS operator *(const double& a, const RotationMatDS& B)
    {
        RotationMatDS tmp;
        tmp.val[0] = B.val[0]*a;
        tmp.val[1] = B.val[1]*a;
        tmp.val[2] = B.val[2]*a;
        tmp.val[3] = B.val[3]*a;
        tmp.val[4] = B.val[4]*a;
        tmp.val[5] = B.val[5]*a;
        tmp.val[6] = B.val[6]*a;
        tmp.val[7] = B.val[7]*a;
        tmp.val[8] = B.val[8]*a;
        return tmp;
    }

    inline friend TranslationDS operator *(const RotationMatDS& A, const TranslationDS& b)
    {
        TranslationDS tmp;

        tmp[0] = A(0,0)*b[0] + A(0,1)*b[1] + A(0,2)*b[2];
        tmp[1] = A(1,0)*b[0] + A(1,1)*b[1] + A(1,2)*b[2];
        tmp[2] = A(2,0)*b[0] + A(2,1)*b[1] + A(2,2)*b[2];

        return tmp;
    }

    inline friend RotationMatDS operator +(const RotationMatDS& A, const RotationMatDS& B)
    {
        RotationMatDS tmp;
        tmp.val[0] = A.val[0]+B.val[0];
        tmp.val[1] = A.val[1]+B.val[1];
        tmp.val[2] = A.val[2]+B.val[2];
        tmp.val[3] = A.val[3]+B.val[3];
        tmp.val[4] = A.val[4]+B.val[4];
        tmp.val[5] = A.val[5]+B.val[5];
        tmp.val[6] = A.val[6]+B.val[6];
        tmp.val[7] = A.val[7]+B.val[7];
        tmp.val[8] = A.val[8]+B.val[8];
        return tmp;
    }

    inline friend RotationMatDS operator -(const RotationMatDS& A, const RotationMatDS& B)
    {
        RotationMatDS tmp;
        tmp.val[0] = A.val[0]-B.val[0];
        tmp.val[1] = A.val[1]-B.val[1];
        tmp.val[2] = A.val[2]-B.val[2];
        tmp.val[3] = A.val[3]-B.val[3];
        tmp.val[4] = A.val[4]-B.val[4];
        tmp.val[5] = A.val[5]-B.val[5];
        tmp.val[6] = A.val[6]-B.val[6];
        tmp.val[7] = A.val[7]-B.val[7];
        tmp.val[8] = A.val[8]-B.val[8];
        return tmp;
    }

    inline RotationMatDS inverse() const
    {
        RotationMatDS tmpRot = *this;
        double tmp;
        tmp = tmpRot.val[1]; tmpRot.val[1]=tmpRot.val[3]; tmpRot.val[3]=tmp;
        tmp = tmpRot.val[2]; tmpRot.val[2]=tmpRot.val[6]; tmpRot.val[6]=tmp;
        tmp = tmpRot.val[5]; tmpRot.val[5]=tmpRot.val[7]; tmpRot.val[7]=tmp;
        return tmpRot;
    }

    inline RotationMatDS transpose() const
    {
        RotationMatDS tmpRot = *this;
        double tmp;
        tmp = tmpRot.val[1]; tmpRot.val[1]=tmpRot.val[3]; tmpRot.val[3]=tmp;
        tmp = tmpRot.val[2]; tmpRot.val[2]=tmpRot.val[6]; tmpRot.val[6]=tmp;
        tmp = tmpRot.val[5]; tmpRot.val[5]=tmpRot.val[7]; tmpRot.val[7]=tmp;
        return tmpRot;
    }

    inline bool isRealRotMat() const
    {
        return (transpose()*(*this) - RotationMatDS()).normal() < MSNH_F32_EPS;
    }

    inline double normal() const
    {
        double final = val[0]*val[0] + val[1]*val[1] + val[2]*val[2] +
                val[3]*val[3] + val[4]*val[4] + val[5]*val[5] +
                val[6]*val[6] + val[7]*val[7] + val[8]*val[8];

        return sqrt(final);
    }

    inline Vector3DS invMul(const Vector3DS& vec) const
    {
        return Vector3DS(
                    val[0]*vec.val[0] + val[3]*vec.val[1] + val[6]*vec.val[2],
                val[1]*vec.val[0] + val[4]*vec.val[1] + val[7]*vec.val[2],
                val[2]*vec.val[0] + val[5]*vec.val[1] + val[8]*vec.val[2]
                );
    }
};
class MsnhNet_API RotationMatFS
{
public :
    float val[9];
    inline RotationMatFS()
    {
        val[0] = 1; val[1] = 0; val[2] = 0;
        val[3] = 0; val[4] = 1; val[5] = 0;
        val[6] = 0; val[7] = 0; val[8] = 1;
    }

    inline RotationMatFS(const std::vector<float>& vec)
    {
        assert(vec.size()==9);
        val[0] = vec[0]; val[1] = vec[1]; val[2] = vec[2];
        val[3] = vec[3]; val[4] = vec[4]; val[5] = vec[5];
        val[6] = vec[6]; val[7] = vec[7]; val[8] = vec[8];
    }

    inline RotationMatFS(const Vector3FS& x, const Vector3FS& y, const Vector3FS& z)
    {
        val[0] = x.val[0]; val[1] = y.val[0]; val[2] = z.val[0];
        val[3] = x.val[1]; val[4] = y.val[1]; val[5] = z.val[1];
        val[6] = x.val[2]; val[7] = y.val[2]; val[8] = z.val[2];
    }

    inline RotationMatFS(const RotationMatFS& vec)
    {
        val[0] = vec.val[0]; val[1] = vec.val[1]; val[2] = vec.val[2];
        val[3] = vec.val[3]; val[4] = vec.val[4]; val[5] = vec.val[5];
        val[6] = vec.val[6]; val[7] = vec.val[7]; val[8] = vec.val[8];
    }

    inline RotationMatFS& operator =(const RotationMatFS& vec)
    {
        if(this!=&vec)
        {
            val[0] = vec.val[0]; val[1] = vec.val[1]; val[2] = vec.val[2];
            val[3] = vec.val[3]; val[4] = vec.val[4]; val[5] = vec.val[5];
            val[6] = vec.val[6]; val[7] = vec.val[7]; val[8] = vec.val[8];
        }
        return *this;
    }

    inline float operator ()(const uint8_t& i, const uint8_t& j) const
    {
        int n = i*3 + j;
        assert(n < 9);
        return val[n];
    }

    inline float &operator ()(const uint8_t& i, const uint8_t& j)
    {
        int n = i*3 + j;
        assert(n < 9);
        return val[n];
    }

    inline void setVal(const std::vector<float>& vec)
    {
        assert(vec.size()==9);
        val[0] = vec[0]; val[1] = vec[1]; val[2] = vec[2];
        val[3] = vec[3]; val[4] = vec[4]; val[5] = vec[5];
        val[6] = vec[6]; val[7] = vec[7]; val[8] = vec[8];
    }

    inline void setX(const Vector3FS& x)
    {
        val[0] = x.val[0];
        val[3] = x.val[1];
        val[6] = x.val[2];
    }

    inline void setY(const Vector3FS& y)
    {
        val[1] = y.val[0];
        val[4] = y.val[1];
        val[7] = y.val[2];
    }

    inline void setZ(const Vector3FS& z)
    {
        val[2] = z.val[0];
        val[5] = z.val[1];
        val[8] = z.val[2];
    }

    inline Vector3FS getX() const
    {
        return Vector3FS(val[0],val[3],val[6]);
    }

    inline Vector3FS getY() const
    {
        return Vector3FS(val[1],val[4],val[7]);
    }

    inline Vector3FS getZ() const
    {
        return Vector3FS(val[2],val[5],val[8]);
    }

    inline friend bool operator ==(const RotationMatFS& A, const RotationMatFS& B)
    {
        if(fabsf(A.val[0]-B.val[0]) < MSNH_F32_EPS && fabsf(A.val[1]-B.val[1]) < MSNH_F32_EPS && fabsf(A.val[2]-B.val[2]) < MSNH_F32_EPS &&
                fabsf(A.val[3]-B.val[3]) < MSNH_F32_EPS && fabsf(A.val[4]-B.val[4]) < MSNH_F32_EPS && fabsf(A.val[5]-B.val[5]) < MSNH_F32_EPS &&
                fabsf(A.val[6]-B.val[6]) < MSNH_F32_EPS && fabsf(A.val[7]-B.val[7]) < MSNH_F32_EPS && fabsf(A.val[8]-B.val[8]) < MSNH_F32_EPS)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    inline friend bool operator !=(const RotationMatFS& A, const RotationMatFS& B)
    {
        if(fabsf(A.val[0]-B.val[0]) < MSNH_F32_EPS && fabsf(A.val[1]-B.val[1]) < MSNH_F32_EPS && fabsf(A.val[2]-B.val[2]) < MSNH_F32_EPS &&
                fabsf(A.val[3]-B.val[3]) < MSNH_F32_EPS && fabsf(A.val[4]-B.val[4]) < MSNH_F32_EPS && fabsf(A.val[5]-B.val[5]) < MSNH_F32_EPS &&
                fabsf(A.val[6]-B.val[6]) < MSNH_F32_EPS && fabsf(A.val[7]-B.val[7]) < MSNH_F32_EPS && fabsf(A.val[8]-B.val[8]) < MSNH_F32_EPS)
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
        if(fabsf(val[0]) < MSNH_F32_EPS && fabsf(val[1]) < MSNH_F32_EPS && fabsf(val[2]) < MSNH_F32_EPS &&
                fabsf(val[3]) < MSNH_F32_EPS && fabsf(val[4]) < MSNH_F32_EPS && fabsf(val[5]) < MSNH_F32_EPS &&
                fabsf(val[6]) < MSNH_F32_EPS && fabsf(val[7]) < MSNH_F32_EPS && fabsf(val[8]) < MSNH_F32_EPS)
        {
            return true;
        }
        return false;
    }

    inline bool closeToEps(const float &eps)
    {
        if(fabsf(val[0]-eps) < MSNH_F32_EPS && fabsf(val[1]-eps) < MSNH_F32_EPS && fabsf(val[2]-eps) < MSNH_F32_EPS &&
                fabsf(val[3]-eps) < MSNH_F32_EPS && fabsf(val[4]-eps) < MSNH_F32_EPS && fabsf(val[5]-eps) < MSNH_F32_EPS &&
                fabsf(val[6]-eps) < MSNH_F32_EPS && fabsf(val[7]-eps) < MSNH_F32_EPS && fabsf(val[8]-eps) < MSNH_F32_EPS)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void print();

    std::string toString() const;

    std::string toHtmlString() const;

    inline friend RotationMatFS operator *(const RotationMatFS& A, const RotationMatFS& B)
    {
        RotationMatFS tmp;
        tmp.val[0] = A.val[0]*B.val[0]+A.val[1]*B.val[3]+A.val[2]*B.val[6];
        tmp.val[1] = A.val[0]*B.val[1]+A.val[1]*B.val[4]+A.val[2]*B.val[7];
        tmp.val[2] = A.val[0]*B.val[2]+A.val[1]*B.val[5]+A.val[2]*B.val[8];
        tmp.val[3] = A.val[3]*B.val[0]+A.val[4]*B.val[3]+A.val[5]*B.val[6];
        tmp.val[4] = A.val[3]*B.val[1]+A.val[4]*B.val[4]+A.val[5]*B.val[7];
        tmp.val[5] = A.val[3]*B.val[2]+A.val[4]*B.val[5]+A.val[5]*B.val[8];
        tmp.val[6] = A.val[6]*B.val[0]+A.val[7]*B.val[3]+A.val[8]*B.val[6];
        tmp.val[7] = A.val[6]*B.val[1]+A.val[7]*B.val[4]+A.val[8]*B.val[7];
        tmp.val[8] = A.val[6]*B.val[2]+A.val[7]*B.val[5]+A.val[8]*B.val[8];
        return tmp;
    }

    inline friend RotationMatFS operator *(const RotationMatFS& A, const float& b)
    {
        RotationMatFS tmp;
        tmp.val[0] = A.val[0]*b;
        tmp.val[1] = A.val[1]*b;
        tmp.val[2] = A.val[2]*b;
        tmp.val[3] = A.val[3]*b;
        tmp.val[4] = A.val[4]*b;
        tmp.val[5] = A.val[5]*b;
        tmp.val[6] = A.val[6]*b;
        tmp.val[7] = A.val[7]*b;
        tmp.val[8] = A.val[8]*b;
        return tmp;
    }

    inline friend RotationMatFS operator *(const float& a, const RotationMatFS& B)
    {
        RotationMatFS tmp;
        tmp.val[0] = B.val[0]*a;
        tmp.val[1] = B.val[1]*a;
        tmp.val[2] = B.val[2]*a;
        tmp.val[3] = B.val[3]*a;
        tmp.val[4] = B.val[4]*a;
        tmp.val[5] = B.val[5]*a;
        tmp.val[6] = B.val[6]*a;
        tmp.val[7] = B.val[7]*a;
        tmp.val[8] = B.val[8]*a;
        return tmp;
    }

    inline friend TranslationFS operator *(const RotationMatFS& A, const TranslationFS& b)
    {
        TranslationFS tmp;

        tmp[0] = A(0,0)*b[0] + A(0,1)*b[1] + A(0,2)*b[2];
        tmp[1] = A(1,0)*b[0] + A(1,1)*b[1] + A(1,2)*b[2];
        tmp[2] = A(2,0)*b[0] + A(2,1)*b[1] + A(2,2)*b[2];

        return tmp;
    }

    inline friend RotationMatFS operator +(const RotationMatFS& A, const RotationMatFS& B)
    {
        RotationMatFS tmp;
        tmp.val[0] = A.val[0]+B.val[0];
        tmp.val[1] = A.val[1]+B.val[1];
        tmp.val[2] = A.val[2]+B.val[2];
        tmp.val[3] = A.val[3]+B.val[3];
        tmp.val[4] = A.val[4]+B.val[4];
        tmp.val[5] = A.val[5]+B.val[5];
        tmp.val[6] = A.val[6]+B.val[6];
        tmp.val[7] = A.val[7]+B.val[7];
        tmp.val[8] = A.val[8]+B.val[8];
        return tmp;
    }

    inline friend RotationMatFS operator -(const RotationMatFS& A, const RotationMatFS& B)
    {
        RotationMatFS tmp;
        tmp.val[0] = A.val[0]-B.val[0];
        tmp.val[1] = A.val[1]-B.val[1];
        tmp.val[2] = A.val[2]-B.val[2];
        tmp.val[3] = A.val[3]-B.val[3];
        tmp.val[4] = A.val[4]-B.val[4];
        tmp.val[5] = A.val[5]-B.val[5];
        tmp.val[6] = A.val[6]-B.val[6];
        tmp.val[7] = A.val[7]-B.val[7];
        tmp.val[8] = A.val[8]-B.val[8];
        return tmp;
    }

    inline RotationMatFS inverse() const
    {
        RotationMatFS tmpRot = *this;
        float tmp;
        tmp = tmpRot.val[1]; tmpRot.val[1]=tmpRot.val[3]; tmpRot.val[3]=tmp;
        tmp = tmpRot.val[2]; tmpRot.val[2]=tmpRot.val[6]; tmpRot.val[6]=tmp;
        tmp = tmpRot.val[5]; tmpRot.val[5]=tmpRot.val[7]; tmpRot.val[7]=tmp;
        return tmpRot;
    }

    inline RotationMatFS transpose() const
    {
        RotationMatFS tmpRot = *this;
        float tmp;
        tmp = tmpRot.val[1]; tmpRot.val[1]=tmpRot.val[3]; tmpRot.val[3]=tmp;
        tmp = tmpRot.val[2]; tmpRot.val[2]=tmpRot.val[6]; tmpRot.val[6]=tmp;
        tmp = tmpRot.val[5]; tmpRot.val[5]=tmpRot.val[7]; tmpRot.val[7]=tmp;
        return tmpRot;
    }

    inline bool isRealRotMat() const
    {
        return (transpose()*(*this) - RotationMatFS()).normal() < MSNH_F32_EPS;
    }

    inline float normal() const
    {
        float final = val[0]*val[0] + val[1]*val[1] + val[2]*val[2] +
                val[3]*val[3] + val[4]*val[4] + val[5]*val[5] +
                val[6]*val[6] + val[7]*val[7] + val[8]*val[8];

        return sqrtf(final);
    }

    inline Vector3FS invMul(const Vector3FS& vec) const
    {
        return Vector3FS(
                    val[0]*vec.val[0] + val[3]*vec.val[1] + val[6]*vec.val[2],
                val[1]*vec.val[0] + val[4]*vec.val[1] + val[7]*vec.val[2],
                val[2]*vec.val[0] + val[5]*vec.val[1] + val[8]*vec.val[2]
                );
    }
};

}
#endif 

