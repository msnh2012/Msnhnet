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
    inline Twist(const LinearVelDS &v, const LinearVelDS &omg):v(v),omg(omg){}
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

    bool closeToEps(const double &eps = MSNH_F64_EPS) const
    {
        if(abs(v[0])<eps && abs(v[1])<eps && abs(v[2])<eps &&
                abs(omg[0])<eps && abs(omg[1])<eps  && abs(omg[2])<eps)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    inline friend Twist operator* (RotationMatDS rot, const Twist& twist)
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

