#ifndef MSNHVECTORS_H
#define MSNHVECTORS_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace Msnhnet
{
template<int N,typename T>
class VectorS
{
public:
    VectorS(){}

    VectorS(const std::vector<T> &val)
    {
        if(val.size()!=N)
        {
            throw Exception(1,"[VectorS]: set val num must equal data num! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        for (int i = 0; i < N; ++i)
        {
            this->_value[i] = val[i];
        }
    }

    VectorS(const VectorS& vec)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i] = vec[i];
        }
    }

    VectorS &operator= (const VectorS &vec)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i] = vec[i];
        }
        return *this;
    }

    VectorS &operator= (const std::vector<T> &val)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i] = val[i];
        }
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
        std::cout<<"{ VectorS: "<<N<<std::endl;
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

        buf<<"{ VectorS: "<<N<<std::endl;
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

        return buf.str();
    }

    inline std::string toHtmlString() const
    {

        std::stringstream buf;

        buf<<"{ VectorS: "<<N<<"<br/>";
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
        return buf.str();
    }

    void setVal(const std::vector<T> &val)
    {
        if(val.size()!=N)
        {
            throw Exception(1,"[VectorS]: set val num must equal data num! \n", __FILE__, __LINE__, __FUNCTION__);
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
            throw Exception(1,"[VectorS]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }

        this->_value[index] = val;
    }

    inline void zero()
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i] = 0;
        }
    }

    inline void reverseSign()
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i] = 0 - this->_value[i];
        }
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
                if(fabs(this->_value[i])>MSNH_F64_EPS)
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

    VectorS normalized() const
    {
        if(!(isF32Vec() || isF64Vec()))
        {
            throw Exception(1, "[VectorS] normalize only f32 and f64 is supported!", __FILE__, __LINE__,__FUNCTION__);
        }

        T len = 0;

        VectorS vec;

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
        else if(isF64Vec())
        {
            if(std::abs(len - 1.0) < MSNH_F64_EPS)
            {
                return *this;
            }

            if(std::abs(len) < MSNH_F64_EPS)
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
            throw Exception(1, "[VectorS] normalize only f32 and f64 is supported!", __FILE__, __LINE__,__FUNCTION__);
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
            if(std::abs(len - 1.0) < MSNH_F64_EPS || std::abs(len) < MSNH_F64_EPS)
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

    inline T operator [](const int &index) const
    {
        if(index > (N-1))
        {
            throw Exception(1,"[VectorS]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        return _value[index];
    }

    inline T &operator [](const int &index)
    {
        if(index > (N-1))
        {
            throw Exception(1,"[VectorS]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
        }
        return _value[index];
    }

    inline friend VectorS operator+ (const VectorS &A, const VectorS &B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] + B[i];
        }
        return tmp;
    }

    inline friend VectorS operator+ (T A, const VectorS &B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A + B[i];
        }
        return tmp;
    }

    inline friend VectorS operator+ (const VectorS &A, T B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] + B;
        }
        return tmp;
    }

    inline friend VectorS operator- (const VectorS &A, const VectorS &B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] - B[i];
        }
        return tmp;
    }

    inline friend VectorS operator- (T A, const VectorS &B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A - B[i];
        }
        return tmp;
    }

    inline friend VectorS operator- (const VectorS &A, T B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] - B;
        }
        return tmp;
    }

    inline friend VectorS operator- (const VectorS &A)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = 0 - A[i];
        }
        return tmp;
    }

    inline friend VectorS operator* (const VectorS &A, const VectorS &B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] * B[i];
        }
        return tmp;
    }

    inline friend VectorS operator* (T A, const VectorS &B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A * B[i];
        }
        return tmp;
    }

    inline friend VectorS operator* (const VectorS &A, T B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] * B;
        }
        return tmp;
    }

    inline friend VectorS operator/ (const VectorS &A, T B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] / B;
        }
        return tmp;
    }

    inline friend VectorS operator/ (const VectorS &A, const VectorS &B)
    {
        VectorS tmp;
        for (int i = 0; i < N; ++i)
        {
            tmp[i] = A[i] / B[i];
        }
        return tmp;
    }

    inline friend bool operator== (const VectorS &A, const VectorS &B)
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

    inline friend bool operator!= (const VectorS &A, const VectorS &B)
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

    inline VectorS &operator +=(const VectorS &A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]+=A[i];
        }
        return *this;
    }

    inline VectorS &operator +=(T A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]+=A;
        }
        return *this;
    }

    inline VectorS &operator -=(const VectorS &A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]-=A[i];
        }
        return *this;
    }

    inline VectorS &operator -=(T A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]-=A;
        }
        return *this;
    }

    inline VectorS &operator *=(const VectorS &A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]*=A[i];
        }
        return *this;
    }

    inline VectorS &operator *=(T A)
    {
        for (int i = 0; i < N; ++i)
        {
            this->_value[i]*=A;
        }
        return *this;
    }

    inline VectorS &operator /=(T A)
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

template<int maxN,typename T>
class VectorXS
{
public:
    int mN    = 0;
    T mValue[maxN];

    inline VectorXS(const int &n)
    {
        assert(n<=maxN);
        mN = n;
        fill(0);
    }

    VectorXS(){}

    inline VectorXS(const std::vector<T> &vec)
    {
        assert(!vec.empty());
        assert((int)vec.size()<=maxN);
        mN = (int)vec.size();

        for (int i = 0; i < mN/4; ++i)
        {
            mValue[(i<<2)+0] = vec[(i<<2)+0];
            mValue[(i<<2)+1] = vec[(i<<2)+1];
            mValue[(i<<2)+2] = vec[(i<<2)+2];
            mValue[(i<<2)+3] = vec[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[i] = vec[i];
        }
    }

    inline VectorXS(const VectorXS& vec)
    {
        this->mN = vec.mN;
        for (int i = 0; i < mN/4; ++i)
        {
            mValue[(i<<2)+0] = vec.mValue[(i<<2)+0];
            mValue[(i<<2)+1] = vec.mValue[(i<<2)+1];
            mValue[(i<<2)+2] = vec.mValue[(i<<2)+2];
            mValue[(i<<2)+3] = vec.mValue[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[i] = vec.mValue[i];
        }
    }

    VectorXS &operator= (const VectorXS& vec)
    {
        if(this!=&vec)
        {
            this->mN = vec.mN;
            for (int i = 0; i < mN/4; ++i)
            {
                mValue[(i<<2)+0] = vec.mValue[(i<<2)+0];
                mValue[(i<<2)+1] = vec.mValue[(i<<2)+1];
                mValue[(i<<2)+2] = vec.mValue[(i<<2)+2];
                mValue[(i<<2)+3] = vec.mValue[(i<<2)+3];
            }

            for (int i = 4*(mN/4); i < mN; ++i)
            {
                mValue[i] = vec.mValue[i];
            }
        }

        return *this;
    }

    VectorXS &operator= (const std::vector<T> &vec)
    {
        assert(!vec.empty());
        assert((int)vec.size()<=maxN);
        mN = (int)vec.size();

        for (int i = 0; i < mN/4; ++i)
        {
            mValue[(i<<2)+0] = vec[(i<<2)+0];
            mValue[(i<<2)+1] = vec[(i<<2)+1];
            mValue[(i<<2)+2] = vec[(i<<2)+2];
            mValue[(i<<2)+3] = vec[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[i] = vec[i];
        }
        return *this;
    }

    inline void fill(const T &value)
    {
        for (int i = 0; i < mN/4; ++i)
        {
            mValue[(i<<2)+0] = value;
            mValue[(i<<2)+1] = value;
            mValue[(i<<2)+2] = value;
            mValue[(i<<2)+3] = value;
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[i] = value;
        }
    }

    void print()
    {
        std::cout<<"{ VectorXS: "<<mN<<std::endl;
        if(isF32Vec())
        {
            for (int i = 0; i < mN; ++i)
            {
                std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<mValue[i]<<" ";
            }
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < mN; ++i)
            {
                std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<mValue[i]<<" ";
            }
        }
        else
        {
            for (int i = 0; i < mN; ++i)
            {
                std::cout<<mValue[i]<<" ";
            }
        }

        std::cout<<";\n}"<<std::endl;
    }

    std::string toString() const
    {

        std::stringstream buf;

        buf<<"{ VectorXS: "<<mN<<std::endl;
        if(isF32Vec())
        {
            for (int i = 0; i < mN; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<mValue[i]<<" ";
            }
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < mN; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<mValue[i]<<" ";
            }
        }
        else
        {
            for (int i = 0; i < mN; ++i)
            {
                buf<<mValue[i]<<" ";
            }
        }

        buf<<";\n}"<<std::endl;

        return buf.str();
    }

    std::string toHtmlString() const
    {

        std::stringstream buf;

        buf<<"{ VectorXS: "<<mN<<"<br/>";
        if(isF32Vec())
        {
            for (int i = 0; i < mN; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<mValue[i]<<" ";
            }
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < mN; ++i)
            {
                buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<mValue[i]<<" ";
            }
        }
        else
        {
            for (int i = 0; i < mN; ++i)
            {
                buf<<mValue[i]<<" ";
            }
        }

        buf<<";\n}"<<"<br/>";

        return buf.str();
    }

    inline void setVal(const std::vector<T> &val)
    {
        assert(val.size()==mN);

        for (int i = 0; i < mN/4; ++i)
        {
            mValue[(i<<2)+0] = val[(i<<2)+0];
            mValue[(i<<2)+1] = val[(i<<2)+1];
            mValue[(i<<2)+2] = val[(i<<2)+2];
            mValue[(i<<2)+3] = val[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[i] = val[i];
        }
    }

    inline void setVal(const int &index, const T &val)
    {
        assert(index<mN);
        this->mValue[index] = val;
    }

    inline void zero()
    {
        for (int i = 0; i < mN/4; ++i)
        {
            mValue[(i<<2)+0] = 0;
            mValue[(i<<2)+1] = 0;
            mValue[(i<<2)+2] = 0;
            mValue[(i<<2)+3] = 0;
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[i] = 0;
        }
    }

    inline void reverseSign()
    {
        for (int i = 0; i < mN/4; ++i)
        {
            mValue[(i<<2)+0] = -mValue[(i<<2)+0];
            mValue[(i<<2)+1] = -mValue[(i<<2)+1];
            mValue[(i<<2)+2] = -mValue[(i<<2)+2];
            mValue[(i<<2)+3] = -mValue[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[i] = -mValue[i];
        }
    }

    inline bool isFuzzyNull() const
    {
        if(isF32Vec()) 

        {
            for (int i = 0; i < mN; ++i)
            {
                if(fabsf(this->mValue[i])>MSNH_F32_EPS)
                {
                    return false;
                }
            }
            return true;
        }
        else if(isF64Vec())
        {
            for (int i = 0; i < mN; ++i)
            {
                if(fabs(this->mValue[i])>MSNH_F64_EPS)
                {
                    return false;
                }
            }
            return true;
        }
        else
        {
            for (int i = 0; i < mN; ++i)
            {
                if(this->mValue[i]>0)
                {
                    return false;
                }
            }
            return true;
        }
    }

    inline bool isFuzzyNull(double eps) const
    {
        for (int i = 0; i < mN; ++i)
        {
            if(fabs(this->mValue[i])>eps)
            {
                return false;
            }
        }
        return true;
    }

    inline bool isNan() const
    {
        for (int i = 0; i < mN; ++i)
        {
            if(std::isnan(static_cast<double>(this->mValue[i])))
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

    inline VectorXS normalized() const
    {
        assert(isF32Vec() || isF64Vec());

        T len = 0;

        VectorXS vec(mN);

        for (int i = 0; i < mN/4; ++i)
        {
            len += this->mValue[(i<<2)+0]*this->mValue[(i<<2)+0];
            len += this->mValue[(i<<2)+1]*this->mValue[(i<<2)+1];
            len += this->mValue[(i<<2)+2]*this->mValue[(i<<2)+2];
            len += this->mValue[(i<<2)+3]*this->mValue[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            len += this->mValue[i]*this->mValue[i];
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
            if(std::abs(len - 1.0) < MSNH_F64_EPS)
            {
                return *this;
            }

            if(std::abs(len) < MSNH_F64_EPS)
            {
                return vec;
            }

            len = sqrt(len);
        }

        for (int i = 0; i < mN; ++i)
        {
            vec[i] = this->mValue[i] / len;
        }

        return vec;
    }

    inline void normalize()
    {
        if(!(isF32Vec() || isF64Vec()))
        {
            throw Exception(1, "[VectorXS] normalize only f32 and f64 is supported!", __FILE__, __LINE__,__FUNCTION__);
        }

        T len = 0;

        for (int i = 0; i < mN/4; ++i)
        {
            len += this->mValue[(i<<2)+0]*this->mValue[(i<<2)+0];
            len += this->mValue[(i<<2)+1]*this->mValue[(i<<2)+1];
            len += this->mValue[(i<<2)+2]*this->mValue[(i<<2)+2];
            len += this->mValue[(i<<2)+3]*this->mValue[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            len += this->mValue[i]*this->mValue[i];
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
            if(std::abs(len - 1.0) < MSNH_F64_EPS || std::abs(len) < MSNH_F64_EPS)
            {
                return;
            }
            len = sqrt(len);
        }

        for (int i = 0; i < mN; ++i)
        {
            this->mValue[i] = this->mValue[i] / len;
        }
    }

    inline double length() const
    {
        double len = 0;
        for (int i = 0; i < mN/4; ++i)
        {
            len += this->mValue[(i<<2)+0]*this->mValue[(i<<2)+0];
            len += this->mValue[(i<<2)+1]*this->mValue[(i<<2)+1];
            len += this->mValue[(i<<2)+2]*this->mValue[(i<<2)+2];
            len += this->mValue[(i<<2)+3]*this->mValue[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            len += this->mValue[i]*this->mValue[i];
        }

        return  sqrt(len);
    }

    inline double lengthSquared() const
    {
        double len = 0;
        for (int i = 0; i < mN/4; ++i)
        {
            len += this->mValue[(i<<2)+0]*this->mValue[(i<<2)+0];
            len += this->mValue[(i<<2)+1]*this->mValue[(i<<2)+1];
            len += this->mValue[(i<<2)+2]*this->mValue[(i<<2)+2];
            len += this->mValue[(i<<2)+3]*this->mValue[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            len += this->mValue[i]*this->mValue[i];
        }
        return  len;
    }

    inline static T dotProduct(const VectorXS &A, const VectorXS &B)
    {
        assert(A.mN == B.mN);

        T finalVal = 0;

        for (int i = 0; i < A.mN/4; ++i)
        {
            finalVal += A[(i<<2)+0]*B[(i<<2)+0];
            finalVal += A[(i<<2)+1]*B[(i<<2)+1];
            finalVal += A[(i<<2)+2]*B[(i<<2)+2];
            finalVal += A[(i<<2)+3]*B[(i<<2)+3];
        }

        for (int i = 4*(A.mN/4); i < A.mN; ++i)
        {
            finalVal += A[i]*B[i];
        }
        return finalVal;
    }

    inline T operator [](const int &index) const
    {
        assert(index < mN);
        return mValue[index];
    }

    inline T &operator [](const int &index)
    {
        assert(index < mN);
        return mValue[index];
    }

    inline T operator ()(const int &index) const
    {
        assert(index < mN);
        return mValue[index];
    }

    inline T &operator ()(const int &index)
    {
        assert(index < mN);
        return mValue[index];
    }

    inline friend VectorXS operator+ (const VectorXS &A, const VectorXS &B)
    {

        assert(A.mN == B.mN);

        VectorXS tmp(A.mN);

        for (int i = 0; i < B.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A[(i<<2)+0]+B[(i<<2)+0];
            tmp[(i<<2)+1] = A[(i<<2)+1]+B[(i<<2)+1];
            tmp[(i<<2)+2] = A[(i<<2)+2]+B[(i<<2)+2];
            tmp[(i<<2)+3] = A[(i<<2)+3]+B[(i<<2)+3];
        }

        for (int i = 4*(B.mN/4); i < B.mN; ++i)
        {
            tmp[i] = A[i]+B[i];
        }

        return tmp;
    }

    inline friend VectorXS operator+ (T A, const VectorXS &B)
    {
        VectorXS tmp(B.mN);
        for (int i = 0; i <B.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A+B[(i<<2)+0];
            tmp[(i<<2)+1] = A+B[(i<<2)+1];
            tmp[(i<<2)+2] = A+B[(i<<2)+2];
            tmp[(i<<2)+3] = A+B[(i<<2)+3];
        }

        for (int i = 4*(B.mN/4); i < B.mN; ++i)
        {
            tmp[i] = A+B[i];
        }
        return tmp;
    }

    inline friend VectorXS operator+ (const VectorXS &A, T B)
    {
        VectorXS tmp(A.mN);
        for (int i = 0; i < B.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A[(i<<2)+0]+B;
            tmp[(i<<2)+1] = A[(i<<2)+1]+B;
            tmp[(i<<2)+2] = A[(i<<2)+2]+B;
            tmp[(i<<2)+3] = A[(i<<2)+3]+B;
        }

        for (int i = 4*(B.mN/4); i < B.mN; ++i)
        {
            tmp[i] = A[i]+B;
        }
        return tmp;
    }

    inline friend VectorXS operator- (const VectorXS &A, const VectorXS &B)
    {

        assert(A.mN == B.mN);

        VectorXS tmp(A.mN);

        for (int i = 0; i < A.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A[(i<<2)+0]-B[(i<<2)+0];
            tmp[(i<<2)+1] = A[(i<<2)+1]-B[(i<<2)+1];
            tmp[(i<<2)+2] = A[(i<<2)+2]-B[(i<<2)+2];
            tmp[(i<<2)+3] = A[(i<<2)+3]-B[(i<<2)+3];
        }

        for (int i = 4*(A.mN/4); i < A.mN; ++i)
        {
            tmp[i] = A[i]-B[i];
        }

        return tmp;
    }

    inline friend VectorXS operator- (T A, const VectorXS &B)
    {
        VectorXS tmp(B.mN);
        for (int i = 0; i < B.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A-B[(i<<2)+0];
            tmp[(i<<2)+1] = A-B[(i<<2)+1];
            tmp[(i<<2)+2] = A-B[(i<<2)+2];
            tmp[(i<<2)+3] = A-B[(i<<2)+3];
        }

        for (int i = 4*(B.mN/4); i < B.mN; ++i)
        {
            tmp[i] = A-B[i];
        }
        return tmp;
    }

    inline friend VectorXS operator- (const VectorXS &A, T B)
    {
        VectorXS tmp(A.mN);
        for (int i = 0; i < A.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A[(i<<2)+0]-B;
            tmp[(i<<2)+1] = A[(i<<2)+1]-B;
            tmp[(i<<2)+2] = A[(i<<2)+2]-B;
            tmp[(i<<2)+3] = A[(i<<2)+3]-B;
        }

        for (int i = 4*(A.mN/4); i < A.mN; ++i)
        {
            tmp[i] = A[i]-B;
        }
        return tmp;
    }

    inline friend VectorXS operator- (const VectorXS &A)
    {
        return 0-A;
    }

    inline friend VectorXS operator* (const VectorXS &A, const VectorXS &B)
    {

        assert(A.mN == B.mN);

        VectorXS tmp(A.mN);

        for (int i = 0; i < A.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A[(i<<2)+0]*B[(i<<2)+0];
            tmp[(i<<2)+1] = A[(i<<2)+1]*B[(i<<2)+1];
            tmp[(i<<2)+2] = A[(i<<2)+2]*B[(i<<2)+2];
            tmp[(i<<2)+3] = A[(i<<2)+3]*B[(i<<2)+3];
        }

        for (int i = 4*(A.mN/4); i < A.mN; ++i)
        {
            tmp[i] = A[i]*B[i];
        }

        return tmp;
    }

    inline friend VectorXS operator* (T A, const VectorXS &B)
    {
        VectorXS tmp(B.mN);
        for (int i = 0; i < B.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A*B[(i<<2)+0];
            tmp[(i<<2)+1] = A*B[(i<<2)+1];
            tmp[(i<<2)+2] = A*B[(i<<2)+2];
            tmp[(i<<2)+3] = A*B[(i<<2)+3];
        }

        for (int i = 4*(B.mN/4); i < B.mN; ++i)
        {
            tmp[i] = A*B[i];
        }
        return tmp;
    }

    inline friend VectorXS operator* (const VectorXS &A, T B)
    {
        VectorXS tmp(A.mN);
        for (int i = 0; i < A.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A[(i<<2)+0]*B;
            tmp[(i<<2)+1] = A[(i<<2)+1]*B;
            tmp[(i<<2)+2] = A[(i<<2)+2]*B;
            tmp[(i<<2)+3] = A[(i<<2)+3]*B;
        }

        for (int i = 4*(A.mN/4); i < A.mN; ++i)
        {
            tmp[i] = A[i]*B;
        }
        return tmp;
    }

    inline friend VectorXS operator/ (const VectorXS &A, T B)
    {
        VectorXS tmp(A.mN);
        for (int i = 0; i < A.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A[(i<<2)+0]/B;
            tmp[(i<<2)+1] = A[(i<<2)+1]/B;
            tmp[(i<<2)+2] = A[(i<<2)+2]/B;
            tmp[(i<<2)+3] = A[(i<<2)+3]/B;
        }

        for (int i = 4*(A.mN/4); i < A.mN; ++i)
        {
            tmp[i] = A[i]/B;
        }
        return tmp;
    }

    inline friend VectorXS operator/ (const VectorXS &A, const VectorXS &B)
    {
        assert(A.mN == B.mN);

        VectorXS tmp(A.mN);

        for (int i = 0; i < A.mN/4; ++i)
        {
            tmp[(i<<2)+0] = A[(i<<2)+0]/B[(i<<2)+0];
            tmp[(i<<2)+1] = A[(i<<2)+1]/B[(i<<2)+1];
            tmp[(i<<2)+2] = A[(i<<2)+2]/B[(i<<2)+2];
            tmp[(i<<2)+3] = A[(i<<2)+3]/B[(i<<2)+3];
        }

        for (int i = 4*(A.mN/4); i < A.AmN; ++i)
        {
            tmp[i] = A[i]/B[i];
        }

        return tmp;
    }

    inline friend bool operator== (const VectorXS &A, const VectorXS &B)
    {
        if(A.mN != B.mN)
        {
            return false;
        }

        if(A.isF32Vec())
        {
            for (int i = 0; i < A.mN; ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F32_EPS)
                {
                    return false;
                }
            }
        }
        else if(A.isF64Vec())
        {
            for (int i = 0; i < A.mN; ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F64_EPS)
                {
                    return false;
                }
            }
        }
        else
        {
            for (int i = 0; i < A.mN; ++i)
            {
                if(A[i] != B[i])
                {
                    return false;
                }
            }

        }
        return true;
    }

    inline friend bool operator!= (const VectorXS &A, const VectorXS &B)
    {
        if(A.mN != B.mN)
        {
            return true;
        }

        if(std::is_same<T,float>::value)
        {
            for (int i = 0; i < A.mN; ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F32_EPS)
                {
                    return true;
                }
            }
        }
        else if(std::is_same<T,double>::value)
        {
            for (int i = 0; i < A.mN; ++i)
            {
                if(fabsf(A[i] - B[i])>MSNH_F64_EPS)
                {
                    return true;
                }
            }
        }
        else
        {
            for (int i = 0; i < A.mN; ++i)
            {
                if(A[i] != B[i])
                {
                    return true;
                }
            }

        }
        return false;
    }

    inline VectorXS &operator +=(const VectorXS &A)
    {
        for (int i = 0; i < mN/4; ++i)
        {
            this->mValue[(i<<2)+0] += A[(i<<2)+0];
            this->mValue[(i<<2)+1] += A[(i<<2)+1];
            this->mValue[(i<<2)+2] += A[(i<<2)+2];
            this->mValue[(i<<2)+3] += A[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            this->mValue[i] += A[i];
        }
        return *this;
    }

    inline VectorXS &operator +=(T A)
    {
        for (int i = 0; i < mN/4; ++i)
        {
            this->mValue[(i<<2)+0] += A;
            this->mValue[(i<<2)+1] += A;
            this->mValue[(i<<2)+2] += A;
            this->mValue[(i<<2)+3] += A;
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            this->mValue[i] += A;
        }
        return *this;
    }

    inline VectorXS &operator -=(const VectorXS &A)
    {
        for (int i = 0; i < mN/4; ++i)
        {
            this->mValue[(i<<2)+0] -= A[(i<<2)+0];
            this->mValue[(i<<2)+1] -= A[(i<<2)+1];
            this->mValue[(i<<2)+2] -= A[(i<<2)+2];
            this->mValue[(i<<2)+3] -= A[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            this->mValue[i] -= A[i];
        }
        return *this;
    }

    inline VectorXS &operator -=(T A)
    {
        for (int i = 0; i < mN/4; ++i)
        {
            this->mValue[(i<<2)+0] -= A;
            this->mValue[(i<<2)+1] -= A;
            this->mValue[(i<<2)+2] -= A;
            this->mValue[(i<<2)+3] -= A;
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            this->mValue[i] -= A;
        }
        return *this;
    }

    inline VectorXS &operator *=(const VectorXS &A)
    {
        for (int i = 0; i < mN/4; ++i)
        {
            this->mValue[(i<<2)+0] *= A[(i<<2)+0];
            this->mValue[(i<<2)+1] *= A[(i<<2)+1];
            this->mValue[(i<<2)+2] *= A[(i<<2)+2];
            this->mValue[(i<<2)+3] *= A[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            this->mValue[i] *= A[i];
        }
        return *this;
    }

    inline VectorXS &operator *=(T A)
    {
        for (int i = 0; i < mN/4; ++i)
        {
            this->mValue[(i<<2)+0] *= A;
            this->mValue[(i<<2)+1] *= A;
            this->mValue[(i<<2)+2] *= A;
            this->mValue[(i<<2)+3] *= A;
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            this->mValue[i] *= A;
        }
        return *this;
    }

    inline VectorXS &operator /=(T A)
    {
        for (int i = 0; i < mN/4; ++i)
        {
            this->mValue[(i<<2)+0] /= A;
            this->mValue[(i<<2)+1] /= A;
            this->mValue[(i<<2)+2] /= A;
            this->mValue[(i<<2)+3] /= A;
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            this->mValue[i] /= A;
        }
        return *this;
    }

};

typedef VectorXS<16  ,double> VectorXSDS;
typedef VectorXS<128 ,double> VectorXMDS;
typedef VectorXS<1024,double> VectorXBDS;

typedef VectorXS<16  ,float> VectorXSFS;
typedef VectorXS<128 ,float> VectorXMFS;
typedef VectorXS<1024,float> VectorXBFS;

typedef VectorS<2,double> Vector2DS;
typedef VectorS<4,double> Vector4DS;
typedef VectorS<5,double> Vector5DS;
typedef VectorS<4,double> Vector4DS;
typedef VectorS<6,double> Vector6DS;
typedef VectorS<7,double> Vector7DS;

typedef VectorS<2,float> Vector2FS;
typedef VectorS<4,float> Vector4FS;
typedef VectorS<5,float> Vector5FS;
typedef VectorS<6,float> Vector6FS;
typedef VectorS<7,float> Vector7FS;
}

#endif 

