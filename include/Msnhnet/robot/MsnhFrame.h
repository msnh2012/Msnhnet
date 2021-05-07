#ifndef MSNHFRAME_H
#define MSNHFRAME_H

#include "Msnhnet/cv/MsnhCVMat.h"
#include "Msnhnet/cv/MsnhCVGeometry.h"

namespace  Msnhnet
{
class MsnhNet_API Twist
{
public:
    Twist(){}
    Twist(const Vector3D &v, const Vector3D &omg):v(v),omg(omg){}

    Twist(const Twist &twist); 

    Twist& operator= (const Twist &twist);

    void zero();
    void reverseSign();

    Vector3D v; 

    Vector3D omg;

};
class MsnhNet_API Frame : public Matrix4x4D
{
public:
    Frame(){}
    Frame(const Mat &mat); 

    Frame(Mat&& mat); 

    Frame(const RotationMatD &rotMat);
    Frame(const TranslationD &trans);
    Frame(const RotationMatD &rotMat, const TranslationD &trans);

    Frame fastInvert();

    static Frame SDH(double a,double alpha,double d,double theta);
    static Frame MDH(double a,double alpha,double d,double theta);

    QuaternionD getQuaternion() const;

    Frame& operator= (const Mat &mat);

};
}

#endif 

