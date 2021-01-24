#ifndef MSNHCVGEOMTRY_H
#define MSNHCVGEOMTRY_H

#include "Msnhnet/cv/MsnhCVMat.h"
#include "Msnhnet/cv/MsnhCVMatOp.h"

#ifndef M_PI
#define M_PI 3.14159265453
#endif

#ifndef ROT_EPS
#define ROT_EPS 0.00001
#endif

namespace Msnhnet
{

double deg2rad(const double &val);
float  deg2rad(const float  &val);
double rad2deg(const double &val);
float  rad2deg(const float  &val);

class MsnhNet_API Geometry
{
public:
    static bool isRealRotMat(Mat &R);

    static RotationMatD euler2RotMat(const EulerD &euler, const RotSequence& seq);
    static RotationMatF euler2RotMat(const EulerF &euler, const RotSequence& seq);

    static QuaternionD  euler2Quaternion(EulerD& euler, const RotSequence& seq);
    static QuaternionF  euler2Quaternion(EulerF& euler, const RotSequence& seq);

    static EulerD rotMat2Euler(RotationMatD& rotMat, const RotSequence &seq);
    static EulerF rotMat2Euler(RotationMatF& rotMat, const RotSequence &seq);

    static EulerD quaternion2Euler(QuaternionD& q, const RotSequence& seq);
    static EulerF quaternion2Euler(QuaternionF& q, const RotSequence& seq);

    static QuaternionD  rotMat2Quaternion(RotationMatD& rotMat);
    static QuaternionF  rotMat2Quaternion(RotationMatF& rotMat);

    static RotationMatD quaternion2RotMat(QuaternionD& q);
    static RotationMatF quaternion2RotMat(QuaternionF& q);

    static QuaternionD  rotVec2Quaternion(RotationVecD& rotVec);
    static QuaternionF  rotVec2Quaternion(RotationVecF& rotVec);

    static RotationVecD quaternion2RotVec(QuaternionD& q);
    static RotationVecF quaternion2RotVec(QuaternionF& q);

    static RotationMatD rotVec2RotMat(RotationVecD& rotVec);
    static RotationMatF rotVec2RotMat(RotationVecF& rotVec);

    static RotationVecD rotMat2RotVec(RotationMatD& rotMat);
    static RotationVecF rotMat2RotVec(RotationMatF& rotMat);

    static RotationVecD euler2RotVec(EulerD& euler, const RotSequence& seq);
    static RotationVecF euler2RotVec(EulerF& euler, const RotSequence& seq);

    static EulerD rotVec2Euler(RotationVecD& rotVec, const RotSequence& seq);
    static EulerF rotVec2Euler(RotationVecF& rotVec, const RotSequence& seq);

    static double clamp(const double &val,const double &min,const double &max);
    static float clamp(const float &val,const float &min,const float &max);
};

class MsnhNet_API Matrix4x4D : public Mat_<4,4,double>
{
public:
    Matrix4x4D();

    Matrix4x4D(const Mat &mat); 

    Matrix4x4D(const Matrix4x4D& mat); 

    Matrix4x4D(const RotationMatD& rotMat, const TransformD& trans);

    Matrix4x4D(const std::vector<double> &val);

    Matrix4x4D& operator= (Matrix4x4D &mat);

    Matrix4x4D& operator= (const Mat &mat);

    RotationMatD getRotationMat() const;

    TransformD getTransform() const;

    static Matrix4x4D eye();

    void setRotationMat(const RotationMatD& rotMat);

    void setTransform(const TransformD& trans);

    void translate(const Vector3D& vector);

    void translate(const double &x, const double &y, const double &z);

    void rotate(const double &angle, const double &x, const double &y, const double &z);

    void rotate(const double &angle, const Vector3D& vector);

    void rotate(const EulerD &euler);

    void scale(const double &x, const double &y, const double &z);

    void scale(const Vector3D& vector);

    void perspective(const double &verticalAngle, const double &aspectRatio,
                      const double &nearPlane, const double &farPlane);

    void ortho(const double &left, const double &right, const double &bottom, const double &top,
                const double &nearPlane, const double &farPlane);

    void lookAt(const Vector3D &eye, const Vector3D &center, const Vector3D &up);
};

class MsnhNet_API Matrix4x4F : public Mat_<4,4,float>
{
public:
    Matrix4x4F();

    Matrix4x4F(const Mat &mat); 

    Matrix4x4F(const Matrix4x4F& mat); 

    Matrix4x4F(const RotationMatF& rotMat, const TransformF& trans);

    Matrix4x4F(const std::vector<float> &val);

    Matrix4x4F& operator= (Matrix4x4F &mat);

    Matrix4x4F& operator= (const Mat &mat);

    RotationMatF getRotationMat() const;

    TransformF getTransform() const;

    static Matrix4x4F eye();

    void setRotationMat(const RotationMatF& rotMat);

    void setTransform(const TransformF& trans);

    void translate(const Vector3F& vector);

    void translate(const float &x, const float &y, const float &z);

    void rotate(const float &angle, const float &x, const float &y, const float &z);

    void rotate(const float &angle, const Vector3F& vector);

    void rotate(const EulerF &euler);

    void scale(const float &x, const float &y, const float &z);

    void scale(const Vector3F& vector);

    void perspective(const float &verticalAngle, const float &aspectRatio,
                      const float &nearPlane, const float &farPlane);

    void ortho(const float &left, const float &right, const float &bottom, const float &top,
                const float &nearPlane, const float &farPlane);

    void lookAt(const Vector3F &eye, const Vector3F &center, const Vector3F &up);
};

}

#endif 

