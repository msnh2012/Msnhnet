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

class MsnhNet_API Frame
{
public:
    RotationMatDS rotMat;
    Vector3DS trans;

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

    inline Frame operator =(const Frame& frame)
    {
        if(this!=&frame)
        {
            this->rotMat = frame.rotMat;
            this->trans  = frame.trans;
        }
        return *this;
    }

    inline void translate(const TranslationDS& vector)
    {
        trans[0] += vector[0];
        trans[1] += vector[1];
        trans[2] += vector[2];
    }

    inline void translate(const double &x, const double &y, const double &z)
    {
        trans[0] += x;
        trans[1] += y;
        trans[2] += z;
    }

    inline void rotate(const double &angleInRad, const double &x, const double &y, const double &z)
    {
        rotate(angleInRad,Vector3DS(x,y,z));
    }

    inline void rotate(const double &angleInRad, const Vector3DS& vector)
    {
        Vector3DS vec = vector;
        vec.normalize();
        const double x = vec[0];
        const double y = vec[1];
        const double z = vec[2];

        rotMat = GeometryS::euler2RotMat(EulerDS(x*angleInRad,y*angleInRad,z*angleInRad),RotSequence::ROT_ZYX);
    }

    inline void rotate(const EulerDS &euler)
    {
        rotMat = GeometryS::euler2RotMat(euler,RotSequence::ROT_ZYX);
    }

    inline void rotate(const QuaternionDS &quat)
    {
        rotMat = GeometryS::quaternion2RotMat(quat);
    }

    inline Frame invert() const
    {
        Frame tmp;

        tmp.rotMat = rotMat.inverse();
        tmp.trans  = rotMat.invMul(trans*-1);

        return tmp;
    }

    inline friend Frame operator *(const Frame& A, const Frame& B)
    {
        return Frame(A.rotMat*B.rotMat, A.rotMat*B.trans+A.trans);
    }

    void print();
    string toString() const;
    string toHtmlString() const;

    static Frame SDH(double a,double alpha,double d,double theta);
    static Frame MDH(double a,double alpha,double d,double theta);

    static Twist diff(const Frame &base2A, const Frame &base2B);
    static Twist diffRelative(const Frame &base2A, const Frame &base2B);
};
}

#endif 

