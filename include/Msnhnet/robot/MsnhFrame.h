#ifndef MSNHFRAME_H
#define MSNHFRAME_H

#include "Msnhnet/cv/MsnhCVGeometry.h"
#include "Msnhnet/math/MsnhMath.h"

namespace  Msnhnet
{

class MsnhNet_API Twist
{
public:
    LinearVelDS v; 

    AngularVelDS omg;

    Twist(){}
    inline Twist(const LinearVelDS &v, const AngularVelDS &omg):v(v),omg(omg){}
    inline Twist(const Twist &twist) 

    {
        this->v     = twist.v;
        this->omg   = twist.omg;
    }

    inline Twist& operator= (const Twist &twist)
    {
        if(this!=&twist)
        {
            this->omg = twist.omg;
            this->v   = twist.v;
        }
        return *this;
    }

    inline Twist refPoint(const Vector3DS& vBaseAB) const
    {
        return Twist(this->v + Vector3DS::crossProduct(this->omg,vBaseAB), this->omg);
    }

    inline double length()
    {
        return sqrt(v[0]*v[0]+
                    v[1]*v[1]+
                    v[2]*v[2]+
                    omg[0]*omg[0]+
                    omg[1]*omg[1]+
                    omg[2]*omg[2]
                );
    }

    bool closeToEps(const double &eps = MSNH_F64_EPS) const
    {
        if(std::abs(v[0])<eps && std::abs(v[1])<eps && std::abs(v[2])<eps &&
                std::abs(omg[0])<eps && std::abs(omg[1])<eps  && std::abs(omg[2])<eps)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    inline friend Twist operator* (const Twist& A, const Twist& B)
    {
        Twist C;
        C.v[0] = A.v[0]*B.v[0];
        C.v[1] = A.v[1]*B.v[1];
        C.v[2] = A.v[2]*B.v[2];

        C.omg[0] = A.omg[0]*B.omg[0];
        C.omg[1] = A.omg[1]*B.omg[1];
        C.omg[2] = A.omg[2]*B.omg[2];

        return C;
    }

    inline friend Twist operator* (const RotationMatDS& rot, const Twist& twist)
    {
        return Twist(rot*twist.v, rot*twist.omg);
    }

    inline double operator [](const uint8_t& i) const
    {
        if(i>2)
        {
            return omg[i-3];
        }
        else
        {
            return v[i];
        }
    }

    void print();

    MatSDS toMat();

    MatSDS toDiagMat();

    VectorXSDS toVec();
};

class MsnhNet_API Frame : public HomTransMatDS
{
public:
    Frame(){}
    inline Frame(const RotationMatDS& rotMat, const Vector3DS& trans)
    {
        this->rotMat = rotMat;
        this->trans  = trans;
    }

    inline Frame(const RotationMatDS& rotMat)
    {
        this->rotMat = rotMat;
    }

    inline Frame(const Vector3DS& trans)
    {
        this->trans  = trans;
    }

    inline Frame(const Frame& frame)
    {
        this->rotMat = frame.rotMat;
        this->trans  = frame.trans;
    }

    inline Frame(const HomTransMatDS& homTransMat)
    {
        this->rotMat = homTransMat.rotMat;
        this->trans  = homTransMat.trans;
    }

    inline Frame operator =(const Frame& frame)
    {
        if(this!=&frame)
        {
            this->rotMat = frame.rotMat;
            this->trans  = frame.trans;
        }
        return *this;
    }

    inline Frame operator =(const HomTransMatDS& homTransMat)
    {
        this->rotMat = homTransMat.rotMat;
        this->trans  = homTransMat.trans;
        return *this;
    }

    static Frame SDH(double a,double alpha,double d,double theta);
    static Frame MDH(double a,double alpha,double d,double theta);

    static Twist diff(const Frame &base2A, const Frame &base2B);
    static Twist diffRelative(const Frame &base2A, const Frame &base2B);
};
}

#endif 

