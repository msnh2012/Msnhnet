#ifndef MSNHQUATERNIONS_H
#define MSNHQUATERNIONS_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

namespace Msnhnet
{
class MsnhNet_API QuaternionDS
{
public:
    double q0;
    double q1;
    double q2;
    double q3;

    inline QuaternionDS(){q0 = q1 = q2 = q3 =0;}

    inline QuaternionDS(const double& q0, const double& q1, const double& q2, const double& q3)
    {
        this->q0 = q0;
        this->q1 = q1;
        this->q2 = q2;
        this->q3 = q3;
    }

    inline QuaternionDS(const QuaternionDS &q)
    {
        q0 = q.q0;
        q1 = q.q1;
        q2 = q.q2;
        q3 = q.q3;
    }

    inline QuaternionDS& operator=(const QuaternionDS& q)
    {
        if(this!=&q)
        {
            q0 = q.q0;
            q1 = q.q1;
            q2 = q.q2;
            q3 = q.q3;
        }
        return *this;
    }

    inline void setVal(const double& q0, const double& q1, const double& q2, const double& q3)
    {
        this->q0 = q0;
        this->q1 = q1;
        this->q2 = q2;
        this->q3 = q3;
    }

    inline double mod() const
    {
        return sqrt(q0*q0 + q1*q1 + q2*q2 + q2*q2);
    }

    inline QuaternionDS invert() const
    {
        double tmp = mod();
        return QuaternionDS(q0/tmp, q1/tmp, q2/tmp, q3/tmp);
    }

    void print();

    std::string toString();

    std::string toHtmlString();

    inline bool operator== (const QuaternionDS& q)
    {
        if(fabs(q0-q.q0)<MSNH_F64_EPS&&
                fabs(q1-q.q1)<MSNH_F64_EPS&&
                fabs(q2-q.q2)<MSNH_F64_EPS&&
                fabs(q3-q.q3)<MSNH_F64_EPS)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    inline bool operator!= (const QuaternionDS& q)
    {
        if(fabs(q0-q.q0)<MSNH_F64_EPS&&
                fabs(q1-q.q1)<MSNH_F64_EPS&&
                fabs(q2-q.q2)<MSNH_F64_EPS&&
                fabs(q3-q.q3)<MSNH_F64_EPS)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    inline friend QuaternionDS operator- (const QuaternionDS &A, const QuaternionDS &B)
    {
        return QuaternionDS(A.q0-B.q0,
                            A.q1-B.q1,
                            A.q2-B.q2,
                            A.q3-B.q3);
    }

    inline friend QuaternionDS operator+ (const QuaternionDS &A, const QuaternionDS &B)
    {
        return QuaternionDS(A.q0+B.q0,
                            A.q1+B.q1,
                            A.q2+B.q2,
                            A.q3+B.q3);
    }

    inline friend QuaternionDS operator* (const QuaternionDS &A, const QuaternionDS &B)
    {
        return QuaternionDS(
                    A.q0*B.q0-A.q1*B.q1-A.q2*B.q2-A.q3*B.q3,
                    A.q0*B.q1+A.q1*B.q0+A.q2*B.q3-A.q3*B.q2,
                    A.q0*B.q2-A.q1*B.q3+A.q2*B.q0+A.q3*B.q1,
                    A.q0*B.q3+A.q1*B.q2-A.q2*B.q1+A.q3*B.q0
                    );
    }

    inline friend QuaternionDS operator/ (const QuaternionDS &A, const QuaternionDS &B)
    {
        return A*B.invert();
    }
};

class MsnhNet_API QuaternionFS
{
public:
    float q0;
    float q1;
    float q2;
    float q3;

    inline QuaternionFS(){q0 = q1 = q2 = q3 =0;}

    inline QuaternionFS(const float& q0, const float& q1, const float& q2, const float& q3)
    {
        this->q0 = q0;
        this->q1 = q1;
        this->q2 = q2;
        this->q3 = q3;
    }

    inline QuaternionFS(const QuaternionFS &q)
    {
        q0 = q.q0;
        q1 = q.q1;
        q2 = q.q2;
        q3 = q.q3;
    }

    inline QuaternionFS& operator=(const QuaternionFS& q)
    {
        if(this!=&q)
        {
            q0 = q.q0;
            q1 = q.q1;
            q2 = q.q2;
            q3 = q.q3;
        }
        return *this;
    }

    inline void setVal(const float& q0, const float& q1, const float& q2, const float& q3)
    {
        this->q0 = q0;
        this->q1 = q1;
        this->q2 = q2;
        this->q3 = q3;
    }

    inline float mod() const
    {
        return sqrtf(q0*q0 + q1*q1 + q2*q2 + q2*q2);
    }

    inline QuaternionFS invert() const
    {
        float tmp = mod();
        return QuaternionFS(q0/tmp, q1/tmp, q2/tmp, q3/tmp);
    }

    void print();

    std::string toString();

    std::string toHtmlString();

    inline bool operator== (const QuaternionFS& q)
    {
        if(fabsf(q0-q.q0)<MSNH_F32_EPS&&
                fabsf(q1-q.q1)<MSNH_F32_EPS&&
                fabsf(q2-q.q2)<MSNH_F32_EPS&&
                fabsf(q3-q.q3)<MSNH_F32_EPS)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    inline bool operator!= (const QuaternionFS& q)
    {
        if(fabsf(q0-q.q0)<MSNH_F32_EPS&&
                fabsf(q1-q.q1)<MSNH_F32_EPS&&
                fabsf(q2-q.q2)<MSNH_F32_EPS&&
                fabsf(q3-q.q3)<MSNH_F32_EPS)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    inline friend QuaternionFS operator- (const QuaternionFS &A, const QuaternionFS &B)
    {
        return QuaternionFS(A.q0-B.q0,
                            A.q1-B.q1,
                            A.q2-B.q2,
                            A.q3-B.q3);
    }

    inline friend QuaternionFS operator+ (const QuaternionFS &A, const QuaternionFS &B)
    {
        return QuaternionFS(A.q0+B.q0,
                            A.q1+B.q1,
                            A.q2+B.q2,
                            A.q3+B.q3);
    }

    inline friend QuaternionFS operator* (const QuaternionFS &A, const QuaternionFS &B)
    {
        return QuaternionFS(
                    A.q0*B.q0-A.q1*B.q1-A.q2*B.q2-A.q3*B.q3,
                    A.q0*B.q1+A.q1*B.q0+A.q2*B.q3-A.q3*B.q2,
                    A.q0*B.q2-A.q1*B.q3+A.q2*B.q0+A.q3*B.q1,
                    A.q0*B.q3+A.q1*B.q2-A.q2*B.q1+A.q3*B.q0
                    );
    }

    inline friend QuaternionFS operator/ (const QuaternionFS &A, const QuaternionFS &B)
    {
        return A*B.invert();
    }
};

}

#endif 

