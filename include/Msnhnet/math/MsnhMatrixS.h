#ifndef MSNHMATRIXS_H
#define MSNHMATRIXS_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhGemm.h"
#include "Msnhnet/math/MsnhVectorS.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace Msnhnet
{

template<int maxH, int maxW, typename T>
class MatS
{
public:
    int mWidth    = 1;
    int mHeight   = 1;
    T mValue[maxH*maxW];

    MatS(){}

    MatS(const int& w, const int &h)
    {
        assert(w<=maxW && h<=maxH);
        mHeight = h;
        mWidth  = w;
        fill(0);
    }

    MatS(const int& w, const int &h, const std::vector<T> &data)
    {
        assert(w<=maxW && h<=maxH);
        assert(data.size()==(w*h));

        mHeight = h;
        mWidth  = w;

        int mN = dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mValue[(i<<2)+0] = data[(i<<2)+0];
            mValue[(i<<2)+1] = data[(i<<2)+1];
            mValue[(i<<2)+2] = data[(i<<2)+2];
            mValue[(i<<2)+3] = data[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[i] = data[i];
        }
    }

    MatS(const MatS& mat)
    {
        this->mHeight = mat.mHeight;
        this->mWidth  = mat.mWidth;
        int mN = dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mValue[(i<<2)+0] = mat.mValue[(i<<2)+0];
            mValue[(i<<2)+1] = mat.mValue[(i<<2)+1];
            mValue[(i<<2)+2] = mat.mValue[(i<<2)+2];
            mValue[(i<<2)+3] = mat.mValue[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[i] = mat.mValue[i];
        }
    }

    MatS &operator= (const MatS& mat)
    {
        if(this!=&mat)
        {
            this->mHeight = mat.mHeight;
            this->mWidth  = mat.mWidth;
            int mN = dataNum();

            for (int i = 0; i < mN/4; ++i)
            {
                mValue[(i<<2)+0] = mat.mValue[(i<<2)+0];
                mValue[(i<<2)+1] = mat.mValue[(i<<2)+1];
                mValue[(i<<2)+2] = mat.mValue[(i<<2)+2];
                mValue[(i<<2)+3] = mat.mValue[(i<<2)+3];
            }

            for (int i = 4*(mN/4); i < mN; ++i)
            {
                mValue[i] = mat.mValue[i];
            }
        }
        return *this;
    }

    inline static MatS eye(const int &num)
    {
        assert(num<=maxH && num<=maxW);

        MatS mEye = MatS(num,num);
        for (int i = 0; i < num; ++i)
        {
            mEye(i,i) = 1;
        }

        return mEye;
    }

    inline static MatS dense(const int &w, const int &h, const T &t)
    {
        assert(h<=maxH && w<=maxW);

        MatS tmp = MatS(w,h);
        tmp.fill(t);
        return tmp;
    }

    inline static MatS diag(const int &num, const T &t)
    {
        assert(num<=maxH && num<=maxW);

        MatS tmp = MatS(num,num);
        for (int i = 0; i < num; ++i)
        {
            tmp(i,i) = t;
        }

        return tmp;
    }

    inline static MatS random(const int &w, const int &h)
    {
        assert(h<=maxH && w<=maxW);

        MatS tmp = MatS(w,h);

        int mN = tmp.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            tmp.mValue[(i<<2)+0] = randUniform((T)(-1),(T)(1));
            tmp.mValue[(i<<2)+1] = randUniform((T)(-1),(T)(1));
            tmp.mValue[(i<<2)+2] = randUniform((T)(-1),(T)(1));
            tmp.mValue[(i<<2)+3] = randUniform((T)(-1),(T)(1));
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            tmp.mValue[i] = randUniform((T)(-1),(T)(1));
        }

        return tmp;
    }

    inline static MatS randomDiag(const int &num)
    {
        assert(num<=maxH && num<=maxW);

        MatS tmp = MatS(num,num);
        for (int i = 0; i < num; ++i)
        {
            tmp(i,i) = randUniform((T)(-1),(T)(1));
        }

        return tmp;
    }

    inline static T randUniform(T min, T max)
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

    inline int dataNum() const
    {
        return mHeight*mWidth;
    }

    inline void fill(const T &value)
    {
        int mN = dataNum();

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

    inline void setVal(const std::vector<T> &val)
    {
        assert(dataNum()==val.size());

        int mN = dataNum();

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

    inline void zero()
    {
        fill(0);
    }

    inline MatS transpose() const
    {
        MatS tmpMat(mHeight,mWidth);

#ifdef USE_OMP
        uint64_t dataLen   = this->mHeight*this->mWidth;
        uint16_t threadNum = dataLen>MIN_OMP_DATA?OMP_THREAD:1;
#pragma omp parallel for num_threads(threadNum)
#endif
        for (int i = 0; i < this->mHeight; ++i)
        {
            for (int j = 0; j < this->mWidth; ++j)
            {
                tmpMat.mValue[j*mWidth+i] = mValue[i*mWidth + j];
            }
        }

        return tmpMat;
    }

    inline T det() const
    {
        assert(mWidth == mHeight);

        T val = 0;

        if(this->mHeight == 2)
        {
            val = mValue[0]*mValue[3] - mValue[1]*mValue[2];
        }
        else if(this->mHeight == 3)
        {
            val = mValue[0]*mValue[4]*mValue[8] + mValue[1]*mValue[5]*mValue[6] + mValue[2]*mValue[3]*mValue[7]
                    -mValue[0]*mValue[7]*mValue[5] - mValue[3]*mValue[1]*mValue[8] - mValue[2]*mValue[4]*mValue[6];
        }
        else
        {
            MatS tmpMat = *this;

            int k = 0;
            int m = tmpMat.mWidth;
            int p = 1;

            for(int i = 0; i < m; i++ )
            {
                k = i;

                for(int j = i+1; j < m; j++ )
                {

                    if( std::abs(tmpMat.mValue[j*m + i]) > std::abs(tmpMat.mValue[k*m + i]))
                    {
                        k = j;
                    }
                }

                if( std::abs(tmpMat.mValue[k*m + i]) < std::numeric_limits<T>::epsilon() )
                    return 0;

                if( k != i )
                {
                    for(int j = i; j < m; j++ )
                    {
                        std::swap(tmpMat.mValue[i*m + j], tmpMat.mValue[k*m + j]);
                    }
                    p = -p;
                }

                T d = -1/tmpMat.mValue[i*m + i];

                for(int j = i+1; j < m; j++ )
                {
                    T alpha = tmpMat.mValue[j*m + i]*d;

                    if(abs(alpha) < std::numeric_limits<T>::epsilon())
                    {
                        continue;
                    }

                    for( k = i; k < m; k++ )
                    {
                        tmpMat.mValue[j*m + k] += alpha*tmpMat.mValue[i*m + k];
                    }
                }
            }

            val = 1;

            for (int i = 0; i < m; ++i)
            {
                val *= tmpMat.mValue[i*m+i];
            }
            val = val*p;
        }

        return val;
    }

    inline T trace() const
    {
        int tmp = std::min(mWidth,mHeight);
        T tr = 0;
        for (int i = 0; i < tmp; ++i)
        {
            tr += mValue[i*mWidth+i];
        }
        return tr;
    }

    inline std::vector<MatS> LUDecomp(bool outLU) const
    {
        assert(mWidth == mHeight);
        MatS A = *this;
        int m  = A.mWidth;
        int k  = 0;
        MatS B = MatS::eye(m);

        for(int i = 0; i < m; i++ )
        {
            k = i;

            for(int j = i+1; j < m; j++ )
            {

                if( std::abs(A.mValue[j*m + i]) > std::abs(A.mValue[k*m + i]))
                {
                    k = j;
                }
            }

            if( std::abs(A.mValue[k*m + i]) < std::numeric_limits<T>::epsilon() )
                throw Exception(1,"[MatS]: det=0, no invert mat! \n", __FILE__, __LINE__, __FUNCTION__);

            if( k != i )
            {
                for(int j = i; j < m; j++ )
                {
                    std::swap(A.mValue[i*m + j], A.mValue[k*m + j]);
                }

                if(!outLU)
                {
                    for(int j = 0; j < m; j++ )
                    {
                        std::swap(B.mValue[i*m + j], B.mValue[k*m + j]);
                    }
                }

            }

            T d = -1/A.mValue[i*m + i];

            for(int j = i+1; j < m; j++ )
            {
                T alpha = A.mValue[j*m + i]*d;

                if(abs(alpha)<std::numeric_limits<T>::epsilon() )
                {
                    continue;
                }

                if(!outLU)
                {

                    for( k = i+1; k < m; k++ )
                    {
                        A.mValue[j*m + k] += alpha*A.mValue[i*m + k];
                    }

                    for( k = 0; k < m; k++ )
                    {
                        B.mValue[j*m + k] += alpha*B.mValue[i*m + k];
                    }
                }
                else
                {

                    for( k = i; k < m; k++ )
                    {
                        if(k < j)
                            B.mValue[j*m + k] = -d*A.mValue[j*m + k];
                        A.mValue[j*m + k] += alpha*A.mValue[i*m + k];
                    }

                }

            }
            if(!outLU)
                A.mValue[i*m + i] = -d;  

        }

        if(!outLU)
        {
            for(int i = m-1; i >= 0; i-- )
            {
                for(int j = 0; j < m; j++ )
                {
                    T s = B.mValue[i*m + j];
                    for( k = i+1; k < m; k++ )
                    {
                        s -= A.mValue[i*m + k]*B.mValue[k*m + j];
                    }
                    B.mValue[i*m + j] = s*A.mValue[i*m + i];
                }
            }
        }

        if(outLU)
        {
            std::vector<MatS> tmpMatVec{B,A};
            return tmpMatVec;
        }
        else
        {
            std::vector<MatS> tmpMatVec{B};
            return tmpMatVec;
        }
    }

    inline std::vector<MatS> choleskyDeComp(bool outChols=true) const
    {
        assert(mWidth == mHeight);

        MatS A   = *this;
        MatS L   = MatS::eye(A.mWidth);
        MatS eye = MatS::eye(A.mWidth);
        int  m   = A.mWidth;

        T s = 0;
        for(int i = 0; i < m; i++ )
        {

            for(int j = 0; j < i; j++ )
            {
                s = A.mValue[j*m + i];
                for(int k = 0; k < j; k++ )
                {
                    s -= A.mValue[i*m + k]*A.mValue[j*m + k];
                }
                A.mValue[i*m + j] = s/A.mValue[j*m + j];
                L.mValue[i*m + j] = s/A.mValue[j*m + j];
            }

            s = A.mValue[i*m + i];
            for(int k = 0; k < i; k++ )
            {
                T t =A.mValue[i*m + k];
                s -= t*t;
            }
            if( s < std::numeric_limits<T>::epsilon() )
                throw Exception(1,"[Mat]: Not a good matrix for cholesky! \n", __FILE__, __LINE__, __FUNCTION__);
            A.mValue[i*m + i] = std::sqrt(s);
            L.mValue[i*m + i] = std::sqrt(s);

            if(!outChols)
            {
                L.mValue[i*m + i] = 1/L.mValue[i*m + i];
            }
        }

        if(!outChols)
        {

            for(int i = 0; i < m; i++ )
            {
                for(int j = 0; j < m; j++ )
                {
                    T s = eye.mValue[i*m + j];
                    for( int k = 0; k < i; k++ )
                    {
                        s -= L.mValue[i*m + k]*eye.mValue[k*m + j];
                    }
                    eye.mValue[i*m + j] = s*L.mValue[i*m + i];
                }
            }

            for(int i = m-1; i >=0; i-- )
            {
                for(int j = 0; j < m; j++ )
                {
                    T s = eye.mValue[i*m + j];
                    for( int k = m-1; k > i; k-- )
                    {
                        s -= L.mValue[k*m + i]*eye.mValue[k*m + j];
                    }
                    eye.mValue[i*m + j] = s*L.mValue[i*m + i];
                }
            }
        }

        if(outChols)
        {
            std::vector<MatS> tmpMatVec{L,L.transpose()};
            return tmpMatVec;
        }
        else
        {

            std::vector<MatS> tmpMatVec{eye};
            return tmpMatVec;
        }
    }

    inline std::vector<MatS> eigen(const bool &sort  = true , const bool& forceCheckSymmetric = false) const
    {
        assert(mWidth == mHeight);

        if(forceCheckSymmetric)
        {
            for (int i = 0; i < mHeight; ++i)
            {
                for (int j = i+1; j < mWidth; ++j)
                {
                    if(mValue[i*mWidth+j]!=mValue[j*mWidth+i])
                    {
                        throw Exception(1,"[MatS]: not a symmetric matrix! \n", __FILE__, __LINE__, __FUNCTION__);
                    }
                }
            }
        }

        int n           = mWidth; 

        MatS eigenvalues(n,1);
        MatS V = MatS::eye(n);
        MatS A = (*this);
        std::vector<int> indR(n, 0);
        std::vector<int> indC(n, 0);
        int maxIters = n*n*30; 

        T maxVal = 0;

        for (int i = 0; i < n; ++i)
        {
            eigenvalues[i] = mValue[i*n+i];
        }

        for (int k = 0; k < n; ++k)
        {
            int maxIdx   = 0;
            int i   = 0;

            if (k < n - 1)
            {
                for (maxIdx = k + 1, maxVal = std::abs(A[n*k + maxIdx]), i = k + 2; i < n; i++)
                {
                    T val = std::abs(A[n*k + i]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        maxIdx = i;
                    }
                }
                indR[k] = maxIdx;
            }

            if (k > 0)
            {
                for (maxIdx = 0, maxVal = std::abs(A[k]), i = 1; i < k; i++)
                {
                    T val = std::abs(A[n*i + k]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        maxIdx = i;
                    }
                }

                indC[k] = maxIdx;
            }
        }

        if (n > 1)
        {
            for (int iters = 0; iters < maxIters; iters++)
            {
                int k   =   0;
                int i   =   0;
                int m   =   0;

                for (k = 0, maxVal = std::abs(A[indR[0]]), i = 1; i < n - 1; i++)
                {
                    T val = std::abs(A[n*i + indR[i]]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        k      = i;
                    }
                }

                int l = indR[k];
                for (i = 1; i < n; i++)
                {
                    T val = std::abs(A[n*indC[i] + i]);
                    if (maxVal < val)
                    {
                        maxVal = val;
                        k = indC[i];
                        l = i;
                    }
                }

                T p = A[n*k + l];

                if (std::abs(p) <= std::numeric_limits<T>::epsilon())
                    break;
                T y = ((eigenvalues[l] - eigenvalues[k])*0.5f);
                T t = std::abs(y) + hypot(p, y);
                T s = hypot(p, t);
                T c = t / s;

                s = p / s;
                t = (p / t)*p;

                if (y < 0)
                {
                    s = -s;
                    t = -t;
                }

                A[n*k + l] = 0;

                eigenvalues[k] -= t;
                eigenvalues[l] += t;

                T a0    =   0;
                T b0    =   0;

#undef rotate
#define rotate(v0, v1) (a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c)

                for (i = 0; i < k; i++)
                {
                    rotate(A[n*i + k], A[n*i + l]);
                }

                for (i = k + 1; i < l; i++)
                {
                    rotate(A[n*k + i], A[n*i + l]);
                }

                for (i = l + 1; i < n; i++)
                {
                    rotate(A[n*k + i], A[n*l + i]);
                }

                for (i = 0; i < n; i++)
                {
                    rotate(V[n*k + i], V[n*l + i]);
                }

#undef rotate

                for (int j = 0; j < 2; j++)
                {
                    int idx = j == 0 ? k : l;
                    if (idx < n - 1)
                    {
                        for (m = idx + 1, maxVal = std::abs(A[n*idx + m]), i = idx + 2; i < n; i++)
                        {
                            T val = std::abs(A[n*idx + i]);

                            if (maxVal < val)
                            {
                                maxVal = val;
                                m = i;
                            }
                        }
                        indR[idx] = m;
                    }
                    if (idx > 0)
                    {
                        for (m = 0, maxVal = std::abs(A[idx]), i = 1; i < idx; i++)
                        {
                            T val = std::abs(A[n*i + idx]);

                            if (maxVal < val)
                            {
                                maxVal = val;
                                m = i;
                            }
                        }
                        indC[idx] = m;
                    }
                }
            }
        }

        if (sort)
        {
            for (int k = 0; k < n - 1; k++)
            {
                int m = k;
                for (int i = k + 1; i < n; i++)
                {
                    if (std::abs(eigenvalues[m]) < std::abs(eigenvalues[i]))
                        m = i;
                }

                if (k != m)
                {
                    std::swap(eigenvalues[m], eigenvalues[k]);

                    for (int i = 0; i < n; i++)
                    {
                        std::swap(V[n*m + i], V[n*k + i]);
                    }
                }
            }
        }

        return std::vector<MatS>{eigenvalues,V.transpose()};
    }

    inline MatS invert() const
    {
        assert(mWidth == mHeight);
        return LUDecomp(false)[0];
    }

    inline MatS pseudoInvert() const
    {

        int m   = mHeight;
        int n   = mWidth;

        auto UDVT = this->svd();

        MatS V   = UDVT[2].transpose();
        MatS UT  = UDVT[0].transpose();

        if(m < n)
        {
            std::swap(m,n);
        }

        MatS Drecip(m,n);

        for (int i = 0; i < n; ++i)
        {
            if(UDVT[1][i] > std::numeric_limits<T>::epsilon())
                Drecip[i*m+i] = 1.0f/UDVT[1][i];
        }

        if(this->mHeight < this->mWidth)
        {
            Drecip = Drecip.transpose();
        }
        return V*Drecip*UT;
    }

    inline static T hypot(T a, T b)
    {
        a = std::abs(a);
        b = std::abs(b);
        if (a > b) {
            b /= a;
            return a*std::sqrt(1 + b*b);
        }
        if (b > 0) {
            a /= b;
            return b*std::sqrt(1 + a*a);
        }
        return 0;
    }

    inline double PYTHAG(double a,double b)
    {
        double at,bt,ct;
        at = std::fabs(a);
        bt = std::fabs(b);
        if (at > bt ) {
            ct=bt/at;
            return at*sqrt(1.0+ct*ct);
        } else {
            if (bt==0)
                return 0.0;
            else {
                ct=at/bt;
                return bt*sqrt(1.0+ct*ct);
            }
        }
    }

    inline double SIGN(double a,double b)
    {
        return (((b) >= 0.0 && std::fabs(b) >MSNH_F64_EPS) ? std::fabs(a) : -std::fabs(a));
    }

    int householderSVD(std::vector<VectorXSDS>& U,VectorXSDS& D,std::vector<VectorXSDS>& V,int maxiter)
    {

        const int height = mHeight;
        const int width  = mWidth;
        VectorXSDS tmp(width);

        int i(-1),its(-1),j(-1),jj(-1),k(-1),nm=0;
        int ppi(0);
        bool flag,maxarg1,maxarg2;
        double anorm(0),c(0),f(0),h(0),s(0),scale(0),x(0),y(0),z(0),g(0);

        for(i=0;i<height;i++)
            for(j=0;j<width;j++)
                U[i](j)=value(j,i);
        if(height>width)
            for(i=height;i<width;i++)
                for(j=0;j<width;j++)
                    U[i](j)=0;

        /* Householder reduction to bidiagonal form. */
        for (i=0;i<width;i++) {
            ppi=i+1;
            tmp(i)=scale*g;
            g=s=scale=0.0;
            if (i<height) {
                for (k=i;k<height;k++) scale += fabs(U[k](i));
                if (scale) {

                    for (k=i;k<height;k++) {
                        U[k](i) /= scale;
                        s += U[k](i)*U[k](i);
                    }
                    f=U[i](i);  

                    g = -SIGN(sqrt(s),f);
                    h=f*g-s;
                    U[i](i)=f-g;
                    for (j=ppi;j<width;j++) {

                        for (s=0.0,k=i;k<height;k++) s += U[k](i)*U[k](j);
                        f=s/h;

                        for (k=i;k<height;k++) U[k](j) += f*U[k](i);
                    }
                    for (k=i;k<height;k++) U[k](i) *= scale;
                }
            }

            D(i)=scale*g;
            g=s=scale=0.0;
            if ((i <height) && (i+1 != width)) {

                for (k=ppi;k<width;k++) scale += fabs(U[i](k));
                if (scale) {
                    for (k=ppi;k<width;k++) {
                        U[i](k) /= scale;
                        s += U[i](k)*U[i](k);
                    }
                    f=U[i](ppi);
                    g = -SIGN(sqrt(s),f);
                    h=f*g-s;
                    U[i](ppi)=f-g;
                    for (k=ppi;k<width;k++) tmp(k)=U[i](k)/h;
                    for (j=ppi;j<height;j++) {
                        for (s=0.0,k=ppi;k<width;k++) s += U[j](k)*U[i](k);
                        for (k=ppi;k<width;k++) U[j](k) += s*tmp(k);
                    }
                    for (k=ppi;k<width;k++) U[i](k) *= scale;
                }
            }
            maxarg1=anorm;
            maxarg2=(fabs(D(i))+fabs(tmp(i)));
            anorm = maxarg1 > maxarg2 ?	maxarg1 : maxarg2;
        }
        /* Accumulation of right-hand transformations. */
        for (i=width-1;i>=0;i--) {
            if (i<width-1) {
                if (g) {
                    for (j=ppi;j<width;j++) V[j](i)=(U[i](j)/U[i](ppi))/g;
                    for (j=ppi;j<width;j++) {
                        for (s=0.0,k=ppi;k<width;k++) s += U[i](k)*V[k](j);
                        for (k=ppi;k<width;k++) V[k](j) += s*V[k](i);
                    }
                }
                for (j=ppi;j<width;j++) V[i](j)=V[j](i)=0.0;
            }
            V[i](i)=1.0;
            g=tmp(i);
            ppi=i;
        }
        /* Accumulation of left-hand transformations. */
        for (i=width-1<height-1 ? width-1:height-1;i>=0;i--) {
            ppi=i+1;
            g=D(i);
            for (j=ppi;j<width;j++) U[i](j)=0.0;
            if (g) {
                g=1.0/g;
                for (j=ppi;j<width;j++) {
                    for (s=0.0,k=ppi;k<height;k++) s += U[k](i)*U[k](j);
                    f=(s/U[i](i))*g;
                    for (k=i;k<height;k++) U[k](j) += f*U[k](i);
                }
                for (j=i;j<height;j++) U[j](i) *= g;
            } else {
                for (j=i;j<height;j++) U[j](i)=0.0;
            }
            ++U[i](i);
        }

        /* Diagonalization of the bidiagonal form. */
        for (k=width-1;k>=0;k--) { /* Loop over singular values. */
            for (its=1;its<=maxiter;its++) {  /* Loop over allowed iterations. */
                flag=true;
                for (ppi=k;ppi>=0;ppi--) {  /* Test for splitting. */
                    nm=ppi-1;             /* Note that tmp[1] is always zero. */
                    if ((fabs(tmp(ppi))+anorm) == anorm) {
                        flag=false;
                        break;
                    }
                    if ((fabs(D(nm)+anorm) == anorm)) break;
                }
                if (flag) {
                    c=0.0;           /* Cancellation of tmp[l], if l>1: */
                    s=1.0;
                    for (i=ppi;i<=k;i++) {
                        f=s*tmp(i);
                        tmp(i)=c*tmp(i);
                        if ((fabs(f)+anorm) == anorm) break;
                        g=D(i);
                        h=PYTHAG(f,g);
                        D(i)=h;
                        h=1.0/h;
                        c=g*h;
                        s=(-f*h);
                        for (j=0;j<height;j++) {
                            y=U[j](nm);
                            z=U[j](i);
                            U[j](nm)=y*c+z*s;
                            U[j](i)=z*c-y*s;
                        }
                    }
                }
                z=D(k);

                if (ppi == k)
                {       /* Convergence. */
                    if (z < 0.0)
                    {   /* Singular value is made nonnegative. */

                        if(std::fabs(z) > MSNH_F64_EPS)
                        {
                            D(k) = -z;
                            for (j=0;j<width;j++) V[j](k)=-V[j](k);
                        }
                    }
                    break;
                }
                x=D(ppi);            /* Shift from bottom 2-by-2 minor: */
                nm=k-1;
                y=D(nm);
                g=tmp(nm);
                h=tmp(k);
                f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);

                g=PYTHAG(f,1.0);
                f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;

                /* Next QR transformation: */
                c=s=1.0;
                for (j=ppi;j<=nm;j++) {
                    i=j+1;
                    g=tmp(i);
                    y=D(i);
                    h=s*g;
                    g=c*g;
                    z=PYTHAG(f,h);
                    tmp(j)=z;
                    c=f/z;
                    s=h/z;
                    f=x*c+g*s;
                    g=g*c-x*s;
                    h=y*s;
                    y=y*c;
                    for (jj=0;jj<width;jj++) {
                        x=V[jj](j);
                        z=V[jj](i);
                        V[jj](j)=x*c+z*s;
                        V[jj](i)=z*c-x*s;
                    }
                    z=PYTHAG(f,h);
                    D(j)=z;
                    if (z) {
                        z=1.0/z;
                        c=f*z;
                        s=h*z;
                    }
                    f=(c*g)+(s*y);
                    x=(c*y)-(s*g);
                    for (jj=0;jj<height;jj++) {
                        y=U[jj](j);
                        z=U[jj](i);
                        U[jj](j)=y*c+z*s;
                        U[jj](i)=z*c-y*s;
                    }
                }
                tmp(ppi)=0.0;
                tmp(k)=f;
                D(k)=x;
            }
        }
        if (its == maxiter)
            return (-2);
        else
            return (0);
    }

    inline void jacobiSVD(MatS &At, MatS &_W, MatS &Vt) const
    {

        double minval = FLT_MIN;
        T eps = (T)(FLT_EPSILON * 2);
        const int m = At.mWidth;  

        const int n = _W.mHeight; 

        const int n1 = m; 

        std::vector<double> W(n, 0);

        Vt = MatS::eye(n);

        for (int i = 0; i < n; i++)
        {
            double sd = 0;
            for (int k = 0; k < m; k++)
            {
                T t = At.mValue[i*m+k];
                sd += (double)t*t;
            }
            W[i] = sd;
        }

        int maxIter = std::max(m, 30);

        for (int iter = 0; iter < maxIter; iter++)
        {
            bool changed = false;

            T c =   0;
            T s =   0;

            for (int i = 0; i < n - 1; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    T *Ai = At.mValue + i*m;
                    T *Aj = At.mValue + j*m;

                    double a = W[i], p = 0, b = W[j];

                    for (int k = 0; k < m; k++)
                        p += (double)Ai[k] * Aj[k];

                    if (std::fabs(p) <= eps * std::sqrt((double)a*b))
                        continue;

                    p *= 2;
                    double beta = a - b, gamma = hypot((double)p, beta);
                    if (beta < 0) {
                        double delta = (gamma - beta)*0.5;
                        s = (T)std::sqrt(delta / gamma);
                        c = (T)(p / (gamma*s * 2));
                    }
                    else {
                        c = (T)std::sqrt((gamma + beta) / (gamma * 2));
                        s = (T)(p / (gamma*c * 2));
                    }

                    a = b = 0;
                    for (int k = 0; k < m; k++) {
                        T t0 = c*Ai[k] + s*Aj[k];
                        T t1 = -s*Ai[k] + c*Aj[k];
                        Ai[k] = t0; Aj[k] = t1;

                        a += (double)t0*t0; b += (double)t1*t1;
                    }
                    W[i] = a; W[j] = b;

                    changed = true;

                    T *Vi = Vt.mValue + i*n;
                    T *Vj = Vt.mValue + j*n;

                    for (int k = 0; k < n; k++) {
                        T t0 = c*Vi[k] + s*Vj[k];
                        T t1 = -s*Vi[k] + c*Vj[k];
                        Vi[k] = t0; Vj[k] = t1;
                    }
                }
            }

            if (!changed)
                break;
        }

        for (int i = 0; i < n; i++)
        {
            double sd = 0;
            for (int k = 0; k < m; k++)
            {
                T t = At.mValue[i*m+k];
                sd += (double)t*t;
            }
            W[i] = std::sqrt(sd);
        }

        for (int i = 0; i < n - 1; i++)
        {
            int j = i;
            for (int k = i + 1; k < n; k++)
            {
                if (W[j] < W[k])
                    j = k;
            }

            if (i != j)
            {
                std::swap(W[i], W[j]);

                for (int k = 0; k < m; k++)
                {
                    std::swap(At.mValue[i*m+k], At.mValue[j*m+k]);
                }

                for (int k = 0; k < n; k++)
                {
                    std::swap(Vt.mValue[i*n+k], Vt.mValue[j*n+k]);
                }
            }
        }

        for (int i = 0; i < n; i++)
        {
            _W.mValue[i] = (T)W[i];
        }

        srand((unsigned int)time(nullptr));

        for (int i = 0; i < n1; i++)
        {
            double sd = i < n ? W[i] : 0;

            for (int ii = 0; ii < 100 && sd <= minval; ii++)
            {

                const T val0 = (T)(1. / m);
                for (int k = 0; k < m; k++)
                {
                    unsigned int rng = rand() % 4294967295; 

                    T val = (rng & 256) != 0 ? val0 : -val0;
                    At.mValue[i*m+k]= val;
                }

                for (int iter = 0; iter < 2; iter++)
                {
                    for (int j = 0; j < i; j++)
                    {
                        sd = 0;

                        for (int k = 0; k < m; k++)
                        {
                            sd += At.mValue[i*m+k] * At.mValue[j*m+k];
                        }

                        T asum = 0;

                        for (int k = 0; k < m; k++)
                        {
                            T t = (T)(At.mValue[i*m+k]- sd*At.mValue[j*m+k]);
                            At.mValue[i*m+k] = t;
                            asum += std::fabs(t);
                        }
                        asum = asum > eps * 100 ? 1 / asum : 0;

                        for (int k = 0; k < m; k++)
                        {
                            At.mValue[i*m+k] *= asum;
                        }
                    }
                }

                sd = 0;
                for (int k = 0; k < m; k++)
                {
                    T t = At.mValue[i*m+k];
                    sd += (double)t*t;
                }
                sd = std::sqrt(sd);
            }

            T s = (T)(sd > minval ? 1 / sd : 0.);

            for (int k = 0; k < m; ++k)
            {
                At.mValue[i*m+k] *= s;
            }
        }
    }

    inline std::vector<MatS> svd() const
    {
        int n = mWidth; 

        int m = mHeight;

        bool at = false;

        if(m<n)
        {
            at = true;
            std::swap(m, n);
        }

        MatS U(m,m);
        MatS D(1,n);
        MatS Vt(n,n);

        MatS AMat;

        if(!at)
        {
            AMat = this->transpose();
        }
        else
        {
            AMat = (*this);
        }

        MatS AMatNew; 

        if(m == n)
        {
            AMatNew = AMat;
        }
        else
        {
            /*     x x x   ->  x x x
             *     x x x       x x x
             *                 0 0 0
             * */

            AMatNew = MatS(m,m); 

            int mN = AMat.dataNum();

            for (int i = 0; i < mN/4; ++i)
            {
                AMatNew.mValue[(i<<2)+0] = AMat.mValue[(i<<2)+0];
                AMatNew.mValue[(i<<2)+1] = AMat.mValue[(i<<2)+1];
                AMatNew.mValue[(i<<2)+2] = AMat.mValue[(i<<2)+2];
                AMatNew.mValue[(i<<2)+3] = AMat.mValue[(i<<2)+3];
            }

            for (int i = 4*(mN/4); i < mN; ++i)
            {
                AMatNew.mValue[i] = AMat.mValue[i];
            }
        }

        jacobiSVD(AMatNew, D, Vt);

        if(!at)
        {
            U   = AMatNew.transpose();
        }
        else
        {
            U   = Vt.transpose();
            Vt  = AMatNew;
        }
        return std::vector<MatS>({U,D,Vt});
    }

    inline MatS getCol(const unsigned int& col) const
    {
        assert(col < (unsigned int)mWidth);

        MatS tmp(1,mHeight);

        int mN = mHeight;

        for (int i = 0; i < mN/4; ++i)
        {
            tmp[(i<<2)+0] = mValue[((i<<2)+0)*mWidth+col];
            tmp[(i<<2)+1] = mValue[((i<<2)+1)*mWidth+col];
            tmp[(i<<2)+2] = mValue[((i<<2)+2)*mWidth+col];
            tmp[(i<<2)+3] = mValue[((i<<2)+3)*mWidth+col];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            tmp[i] = mValue[i*mWidth+col];
        }
        return tmp;
    }

    inline void setCol(const unsigned int& col,const MatS& mat)
    {
        assert(col < (unsigned int)mWidth);
        assert(mat.mWidth==1 && mat.mHeight==this->mHeight);

        int mN = mHeight;

        for (int i = 0; i < mN/4; ++i)
        {
            mValue[((i<<2)+0)*mWidth+col] = mat[(i<<2)+0];
            mValue[((i<<2)+1)*mWidth+col] = mat[(i<<2)+1];
            mValue[((i<<2)+2)*mWidth+col] = mat[(i<<2)+2];
            mValue[((i<<2)+3)*mWidth+col] = mat[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[i*mWidth+col] = mat[i];
        }
    }

    inline MatS getRow(const unsigned int& row)
    {
        assert(row < (unsigned int)mHeight);

        MatS tmp(mWidth,1);

        int mN   = mWidth;
        int line = row*mWidth;

        for (int i = 0; i < mN/4; ++i)
        {
            tmp[(i<<2)+0] = mValue[line+(i<<2)+0];
            tmp[(i<<2)+1] = mValue[line+(i<<2)+1];
            tmp[(i<<2)+2] = mValue[line+(i<<2)+2];
            tmp[(i<<2)+3] = mValue[line+(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            tmp[i] = mValue[line + i];
        }
        return tmp;
    }

    inline void setRow(const unsigned int& row,const MatS& mat)
    {
        assert(row < (unsigned int)mHeight);
        assert(mat.mHeight==1 && mat.mWidth==mWidth);

        int mN   = mWidth;
        int line = row*mWidth;

        for (int i = 0; i < mN/4; ++i)
        {
            mValue[line+(i<<2)+0] = mat[(i<<2)+0];
            mValue[line+(i<<2)+1] = mat[(i<<2)+1];
            mValue[line+(i<<2)+2] = mat[(i<<2)+2];
            mValue[line+(i<<2)+3] = mat[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mValue[line + i] = mat[i];
        }
    }

    inline VectorXSDS mulVec(const VectorXSDS &vec) const
    {
        assert(mWidth == vec.mN);

        VectorXSDS res(this->mHeight);

        for (int i = 0; i < this->mHeight; ++i)
        {
            T val = 0;
            for (int j = 0; j < this->mWidth; ++j)
            {
                val += mValue[i*this->mWidth+j]*vec[j];
            }

            res[i] = val;
        }

        return res;
    }

    void print()
    {
        if(isF32Mat())
        {
            std::cout<<"{MatFS  width: "<<this->mWidth<<" , height: "<<this->mHeight<<std::endl;
            std::cout<<"    ["<<std::endl;
            for (int i = 0; i < this->mHeight; ++i)
            {
                if(i<19|| (i==this->mHeight-1) )
                {
                    for (int j = 0; j < this->mWidth; ++j)
                    {
                        if(j==0)
                        {
                            std::cout<<"        ";
                        }

                        if(j==19)
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                        }
                        else if(j<19 || j==(this->mWidth-1) )
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<std::setiosflags(std::ios::fixed)<<mValue[i*mWidth + j]<<" ";
                        }
                    }
                    std::cout<<";"<<std::endl;
                }
                else if(i == 20)
                {
                    std::cout<<"        "<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                    std::cout<<";"<<std::endl;
                }
            }
            std::cout<<"    ],"<<std::endl;
        }
        else if(isF64Mat())
        {
            std::cout<<"{MatDS  width: "<<this->mWidth<<" , height: "<<this->mHeight<<std::endl;
            std::cout<<"    ["<<std::endl;
            for (int i = 0; i < this->mHeight; ++i)
            {
                if(i<9|| (i==this->mHeight-1) )
                {
                    for (int j = 0; j < this->mWidth; ++j)
                    {
                        if(j==0)
                        {
                            std::cout<<"        ";
                        }

                        if(j==9)
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                        }
                        else if(j<9 || j==(this->mWidth-1) )
                        {
                            std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(12)<<std::setiosflags(std::ios::fixed)<<mValue[i*mWidth + j]<<" ";
                        }
                    }
                    std::cout<<";"<<std::endl;
                }
                else if(i == 10)
                {
                    std::cout<<"        "<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                    std::cout<<";"<<std::endl;
                }

            }
            std::cout<<"    ],"<<std::endl;
        }
        std::cout<<";\n}"<<std::endl;
    }

    std::string toString()
    {
        std::stringstream buf;
        if(isF32Mat())
        {
            buf<<"{MatFS  width: "<<this->mWidth<<" , height: "<<this->mHeight<<std::endl;
            buf<<"    ["<<std::endl;
            for (int i = 0; i < this->mHeight; ++i)
            {
                if(i<19|| (i==this->mHeight-1) )
                {
                    for (int j = 0; j < this->mWidth; ++j)
                    {
                        if(j==0)
                        {
                            buf<<"        ";
                        }

                        if(j==19)
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                        }
                        else if(j<19 || j==(this->mWidth-1) )
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<std::setiosflags(std::ios::fixed)<<mValue[i*mWidth + j]<<" ";
                        }
                    }
                    buf<<";"<<std::endl;
                }
                else if(i == 20)
                {
                    buf<<"        "<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                    buf<<";"<<std::endl;
                }
            }
            buf<<"    ],"<<std::endl;
        }
        else if(isF64Mat())
        {
            buf<<"{MatDS  width: "<<this->mWidth<<" , height: "<<this->mHeight<<std::endl;
            buf<<"    ["<<std::endl;
            for (int i = 0; i < this->mHeight; ++i)
            {
                if(i<9|| (i==this->mHeight-1) )
                {
                    for (int j = 0; j < this->mWidth; ++j)
                    {
                        if(j==0)
                        {
                            buf<<"        ";
                        }

                        if(j==9)
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                        }
                        else if(j<9 || j==(this->mWidth-1) )
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(12)<<std::setiosflags(std::ios::fixed)<<mValue[i*mWidth + j]<<" ";
                        }
                    }
                    buf<<";"<<std::endl;
                }
                else if(i == 10)
                {
                    buf<<"        "<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                    buf<<";"<<std::endl;
                }

            }
            buf<<"    ],"<<std::endl;
        }
        buf<<";\n}"<<std::endl;
        return buf.str();
    }

    std::string toHtmlString()
    {
        std::stringstream buf;
        if(isF32Mat())
        {
            buf<<"{MatFS  width: "<<this->mWidth<<" , height: "<<this->mHeight<<"<br/>";
            buf<<"    ["<<std::endl;
            for (int i = 0; i < this->mHeight; ++i)
            {
                if(i<19|| (i==this->mHeight-1) )
                {
                    for (int j = 0; j < this->mWidth; ++j)
                    {
                        if(j==0)
                        {
                            buf<<"        ";
                        }

                        if(j==19)
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                        }
                        else if(j<19 || j==(this->mWidth-1) )
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<std::setiosflags(std::ios::fixed)<<mValue[i*mWidth + j]<<" ";
                        }
                    }
                    buf<<";"<<"<br/>";
                }
                else if(i == 20)
                {
                    buf<<"        "<<std::setiosflags(std::ios::left)<<std::setw(6)<<"...";
                    buf<<";"<<"<br/>";
                }
            }
            buf<<"    ],"<<"<br/>";
        }
        else if(isF64Mat())
        {
            buf<<"{MatDS  width: "<<this->mWidth<<" , height: "<<this->mHeight<<std::endl;
            buf<<"    ["<<std::endl;
            for (int i = 0; i < this->mHeight; ++i)
            {
                if(i<9|| (i==this->mHeight-1) )
                {
                    for (int j = 0; j < this->mWidth; ++j)
                    {
                        if(j==0)
                        {
                            buf<<"        ";
                        }

                        if(j==9)
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                        }
                        else if(j<9 || j==(this->mWidth-1) )
                        {
                            buf<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(12)<<std::setiosflags(std::ios::fixed)<<mValue[i*mWidth + j]<<" ";
                        }
                    }
                    buf<<";"<<"<br/>";
                }
                else if(i == 10)
                {
                    buf<<"        "<<std::setiosflags(std::ios::left)<<std::setw(12)<<"...";
                    buf<<";"<<"<br/>";
                }

            }
            buf<<"    ],"<<"<br/>";
        }
        buf<<"}"<<"<br/>"<<"<br/>";
        return buf.str();
    }

    inline bool isF32Mat() const
    {
        return std::is_same<T,float>::value;
    }

    inline bool isF64Mat() const
    {
        return std::is_same<T,double>::value;
    }

    inline T L1() const
    {
        T l1 = 0;

        int mN = dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
           l1 += mValue[(i<<2)+0];
           l1 += mValue[(i<<2)+1];
           l1 += mValue[(i<<2)+2];
           l1 += mValue[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            l1 += mValue[i];
        }

        return l1;
    }

    inline T L2() const
    {
        T l2 = 0;
        int mN = dataNum();
        for (int i = 0; i < mN/4; ++i)
        {
           l2 += mValue[(i<<2)+0]*value[(i<<2)+0];
           l2 += mValue[(i<<2)+1]*value[(i<<2)+1];
           l2 += mValue[(i<<2)+2]*value[(i<<2)+2];
           l2 += mValue[(i<<2)+3]*value[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            l2 += mValue[i]*mValue[i];
        }

        return sqrt(l2);
    }

    inline T LInf() const
    {
        std::vector<T> val;

        for (int i = 0; i < dataNum(); ++i)
        {
            val.push_back(std::abs(mValue[i]));
        }

        return *std::max_element(val.begin(),val.end());
    }

    inline bool isFuzzyNull() const
    {
        if(this->isF32Mat())
        {
            for (size_t i = 0; i < dataNum(); ++i)
            {
                if(fabsf(mValue[i]) > MSNH_F32_EPS)
                {
                    return false;
                }
            }
            return true;
        }
        else
        {
            for (size_t i = 0; i < dataNum(); ++i)
            {
                if(fabs(mValue[i]) > MSNH_F64_EPS)
                {
                    return false;
                }
            }
            return true;
        }
    }

    inline T value(const int &w, const int& h) const
    {
        assert(w<mWidth && h<mHeight);
        return mValue[h*mWidth+w];
    }

    inline T operator ()(const int &w, const int& h) const
    {
        assert(w<mWidth && h<mHeight);
        return mValue[h*mWidth+w];
    }

    inline T& operator ()(const int &w, const int& h)
    {
        assert(w<mWidth && h<mHeight);
        return mValue[h*mWidth+w];
    }

    inline T operator [](const int &index) const
    {
        assert(index < dataNum());
        return mValue[index];
    }

    inline T &operator [](const int &index)
    {
        assert(index < dataNum());
        return mValue[index];
    }

    inline static MatS eleWiseMul(const MatS &A, const MatS &B)
    {
        assert(A.mWidth==B.mWidth && A.mHeight==B.mHeight);

        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = A[(i<<2)+0]*B[(i<<2)+0];
            mat[(i<<2)+1] = A[(i<<2)+1]*B[(i<<2)+1];
            mat[(i<<2)+2] = A[(i<<2)+2]*B[(i<<2)+2];
            mat[(i<<2)+3] = A[(i<<2)+3]*B[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = A[i]*B[i];
        }

        return mat;
    }

    inline static MatS eleWiseDiv(const MatS &A, const MatS &B)
    {
        assert(A.mWidth==B.mWidth && A.mHeight==B.mHeight);

        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = A[(i<<2)+0]/B[(i<<2)+0];
            mat[(i<<2)+1] = A[(i<<2)+1]/B[(i<<2)+1];
            mat[(i<<2)+2] = A[(i<<2)+2]/B[(i<<2)+2];
            mat[(i<<2)+3] = A[(i<<2)+3]/B[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = A[i]/B[i];
        }

        return mat;
    }

    inline friend MatS operator+ (const MatS &A, const MatS &B)
    {
        assert(A.mWidth==B.mWidth && A.mHeight==B.mHeight);

        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = A[(i<<2)+0]+B[(i<<2)+0];
            mat[(i<<2)+1] = A[(i<<2)+1]+B[(i<<2)+1];
            mat[(i<<2)+2] = A[(i<<2)+2]+B[(i<<2)+2];
            mat[(i<<2)+3] = A[(i<<2)+3]+B[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = A[i]+B[i];
        }

        return mat;
    }
    inline friend MatS operator+ (const T &a, const MatS &A)
    {
        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = A[(i<<2)+0]+a;
            mat[(i<<2)+1] = A[(i<<2)+1]+a;
            mat[(i<<2)+2] = A[(i<<2)+2]+a;
            mat[(i<<2)+3] = A[(i<<2)+3]+a;
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = A[i]+a;
        }

        return mat;
    }
    inline friend MatS operator+ (const MatS &A, const T &a)
    {
        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = A[(i<<2)+0]+a;
            mat[(i<<2)+1] = A[(i<<2)+1]+a;
            mat[(i<<2)+2] = A[(i<<2)+2]+a;
            mat[(i<<2)+3] = A[(i<<2)+3]+a;
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = A[i]+a;
        }

        return mat;
    }

    inline friend MatS operator- (const MatS &A, const MatS &B)
    {
        assert(A.mWidth==B.mWidth && A.mHeight==B.mHeight);

        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = A[(i<<2)+0]-B[(i<<2)+0];
            mat[(i<<2)+1] = A[(i<<2)+1]-B[(i<<2)+1];
            mat[(i<<2)+2] = A[(i<<2)+2]-B[(i<<2)+2];
            mat[(i<<2)+3] = A[(i<<2)+3]-B[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = A[i]-B[i];
        }

        return mat;
    }
    inline friend MatS operator- (const MatS &A)
    {
        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = -A[(i<<2)+0];
            mat[(i<<2)+1] = -A[(i<<2)+1];
            mat[(i<<2)+2] = -A[(i<<2)+2];
            mat[(i<<2)+3] = -A[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = -A[i];
        }

        return mat;
    }
    inline friend MatS operator- (const T &a, const MatS &A)
    {
        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = a-A[(i<<2)+0];
            mat[(i<<2)+1] = a-A[(i<<2)+1];
            mat[(i<<2)+2] = a-A[(i<<2)+2];
            mat[(i<<2)+3] = a-A[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = a-A[i];
        }

        return mat;
    }
    inline friend MatS operator- (const MatS &A, const T &a)
    {
        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = A[(i<<2)+0]-a;
            mat[(i<<2)+1] = A[(i<<2)+1]-a;
            mat[(i<<2)+2] = A[(i<<2)+2]-a;
            mat[(i<<2)+3] = A[(i<<2)+3]-a;
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = A[i]-a;
        }

        return mat;
    }

    inline friend MatS operator* (const MatS &A, const MatS &B)
    {
        assert(A.mWidth==B.mHeight);

        MatS mat(B.mWidth, A.mHeight);

        SimdInfo::checkSimd();

        if(A.isF32Mat())
        {
#ifdef USE_X86
            Gemm::cpuGemm(0,0,A.mHeight,B.mWidth,A.mWidth,1,(float*)A.mValue,A.mWidth,(float*)B.mValue,B.mWidth,1,(float*)mat.mValue,mat.mWidth, SimdInfo::supportAVX2);
#else
            Gemm::cpuGemm(0,0,A.mHeight,B.mWidth,A.mWidth,1,(float*)A.mValue,A.mWidth,(float*)B.mValue,B.mWidth,1,(float*)mat.mValue,mat.mWidth, false);
#endif
        }
        else
        {
#ifdef USE_X86
            Gemm::cpuGemm(0,0,A.mHeight,B.mWidth,A.mWidth,1,(double*)A.mValue,A.mWidth,(double*)B.mValue,B.mWidth,1,(double*)mat.mValue,mat.mWidth, SimdInfo::supportAVX2);
#else
            Gemm::cpuGemm(0,0,A.mHeight,B.mWidth,A.mWidth,1,(double*)A.mValue,A.mWidth,(double*)B.mValue,B.mWidth,1,(double*)mat.mValue,mat.mWidth, false);
#endif
        }
        return mat;
    }
    inline friend MatS operator* (const T &a, const MatS &A)
    {
        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = a*A[(i<<2)+0];
            mat[(i<<2)+1] = a*A[(i<<2)+1];
            mat[(i<<2)+2] = a*A[(i<<2)+2];
            mat[(i<<2)+3] = a*A[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = a*A[i];
        }

        return mat;
    }
    inline friend MatS operator* (const MatS &A, const T &a)
    {
        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = a*A[(i<<2)+0];
            mat[(i<<2)+1] = a*A[(i<<2)+1];
            mat[(i<<2)+2] = a*A[(i<<2)+2];
            mat[(i<<2)+3] = a*A[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = a*A[i];
        }

        return mat;
    }

    inline static T dotProduct(const MatS &A, const MatS &B)
    {
        assert(A.mWidth==B.mWidth && A.mHeight==B.mHeight);

        T final = 0;

        int mN = A.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            final += A[(i<<2)+0]*B[(i<<2)+0];
            final += A[(i<<2)+1]*B[(i<<2)+1];
            final += A[(i<<2)+2]*B[(i<<2)+2];
            final += A[(i<<2)+3]*B[(i<<2)+3];
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            final += A[i]*B[i];
        }
        return final;
    }

    inline friend MatS operator/ (const MatS &A, const MatS &B)
    {
        assert(A.mWidth==B.mWidth && A.mHeight==B.mHeight);

        return A*B.invert();
    }
    inline friend MatS operator/ (const T &a, const MatS &A)
    {
        return a*A.invert();
    }
    inline friend MatS operator/ (const MatS &A, const T &a)
    {
        MatS mat(A.mWidth,A.mHeight);

        int mN = mat.dataNum();

        for (int i = 0; i < mN/4; ++i)
        {
            mat[(i<<2)+0] = A[(i<<2)+0]/a;
            mat[(i<<2)+1] = A[(i<<2)+1]/a;
            mat[(i<<2)+2] = A[(i<<2)+2]/a;
            mat[(i<<2)+3] = A[(i<<2)+3]/a;
        }

        for (int i = 4*(mN/4); i < mN; ++i)
        {
            mat[i] = A[i]/a;
        }

        return mat;
    }

    inline friend bool operator == (const MatS &A, const MatS &B)
    {
        if(A.mHeight!=B.mHeight || A.mWidth!=B.mWidth)
        {
            return false;
        }

        int n = A.dataNum();

        for (int i = 0; i < n; ++i)
        {
            if(((T)std::abs(A[i]-B[i]))>std::numeric_limits<T>::epsilon())
            {
                return false;
            }
        }

        return true;
    }

    inline friend bool operator != (const MatS &A, const MatS &B)
    {
        if(A.mHeight!=B.mHeight || A.mWidth!=B.mWidth)
        {
            return true;
        }

        int n = A.dataNum();

        for (int i = 0; i < n; ++i)
        {
            if(((T)std::abs(A[i]-B[i]))>std::numeric_limits<T>::epsilon())
            {
                return true;
            }
        }

        return false;
    }
};

typedef MatS<8,8,double>    MatSDS;
typedef MatS<16,16,double>  MatMDS;

}
#endif 

