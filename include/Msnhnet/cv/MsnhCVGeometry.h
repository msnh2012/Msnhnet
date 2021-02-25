#ifndef MSNHCVGEOMTRY_H
#define MSNHCVGEOMTRY_H

#include "Msnhnet/cv/MsnhCVMat.h"
#include "Msnhnet/cv/MsnhCVVector.h"
#include "Msnhnet/cv/MsnhCVMatOp.h"

#ifndef M_PI
#define M_PI 3.14159265453
#endif

#ifndef ROT_EPS
#define ROT_EPS 0.00001
#endif

namespace Msnhnet
{

MsnhNet_API double deg2rad(const double &val);
MsnhNet_API float  deg2rad(const float  &val);
MsnhNet_API double rad2deg(const double &val);
MsnhNet_API float  rad2deg(const float  &val);

class MsnhNet_API Matrix4x4D : public Mat_<4,4,double>
{
public:
    Matrix4x4D();

    Matrix4x4D(const Mat &mat); 

    Matrix4x4D(const Matrix4x4D& mat); 

    Matrix4x4D(const RotationMatD& rotMat);

    Matrix4x4D(const TranslationD& trans);

    Matrix4x4D(const RotationMatD& rotMat, const TranslationD& trans);

    Matrix4x4D(const std::vector<double> &val);

    Matrix4x4D& operator= (Matrix4x4D &mat);

    Matrix4x4D& operator= (const Mat &mat);

    RotationMatD getRotationMat() const;

    TranslationD getTranslation() const;

    void setRotationMat(const RotationMatD& rotMat);

    void setTranslation(const TranslationD& trans);

    void translate(const Vector3D& vector);

    void translate(const double &x, const double &y, const double &z);

    void rotate(const double &angle, const double &x, const double &y, const double &z);

    void rotate(const double &angle, const Vector3D& vector);

    void rotate(const EulerD &euler);

    void rotate(const QuaternionD &quat);

    void scale(const double &x, const double &y, const double &z);

    void scale(const Vector3D& vector);

    void perspective(const double &verticalAngle, const double &aspectRatio,
                      const double &nearPlane, const double &farPlane);

    void ortho(const double &left, const double &right, const double &bottom, const double &top,
                const double &nearPlane, const double &farPlane);

    void lookAt(const Vector3D &eye, const Vector3D &center, const Vector3D &up);

    Vector3D mulVec3(const Vector3D &vec3);

    Vector4D mulVec4(const Vector4D &vec4);

    Matrix3x3D normalMatrix();

};

class MsnhNet_API Matrix4x4F : public Mat_<4,4,float>
{
public:
    Matrix4x4F();

    Matrix4x4F(const Mat &mat); 

    Matrix4x4F(const Matrix4x4F& mat); 

    Matrix4x4F(const RotationMatF& rotMat, const TranslationF& trans);

    Matrix4x4F(const std::vector<float> &val);

    Matrix4x4F& operator= (Matrix4x4F &mat);

    Matrix4x4F& operator= (const Mat &mat);

    RotationMatF getRotationMat() const;

    TranslationF getTranslation() const;

    void setRotationMat(const RotationMatF& rotMat);

    void setTranslation(const TranslationF& trans);

    void translate(const Vector3F& vector);

    void translate(const float &x, const float &y, const float &z);

    void rotate(const float &angle, const float &x, const float &y, const float &z);

    void rotate(const float &angle, const Vector3F& vector);

    void rotate(const EulerF &euler);

    void rotate(const QuaternionF &quat);

    void scale(const float &x, const float &y, const float &z);

    void scale(const Vector3F& vector);

    void perspective(const float &verticalAngle, const float &aspectRatio,
                      const float &nearPlane, const float &farPlane);

    void ortho(const float &left, const float &right, const float &bottom, const float &top,
                const float &nearPlane, const float &farPlane);

    void lookAt(const Vector3F &eye, const Vector3F &center, const Vector3F &up);

    Vector3F mulVec3(const Vector3F &vec3);

    Vector4F mulVec4(const Vector4F &vec4);

    Matrix3x3F normalMatrix();
};

class MsnhNet_API Geometry
{
public:
    static bool isRealRotMat(Mat &R);

    static RotationMatD euler2RotMat(const EulerD &euler, const RotSequence& seq);
    static RotationMatF euler2RotMat(const EulerF &euler, const RotSequence& seq);

    static QuaternionD  euler2Quaternion(const EulerD& euler, const RotSequence& seq);
    static QuaternionF  euler2Quaternion(const EulerF& euler, const RotSequence& seq);

    static EulerD rotMat2Euler(const RotationMatD& rotMat, const RotSequence &seq);
    static EulerF rotMat2Euler(const RotationMatF& rotMat, const RotSequence &seq);

    static EulerD quaternion2Euler(const QuaternionD& q, const RotSequence& seq);
    static EulerF quaternion2Euler(const QuaternionF& q, const RotSequence& seq);

    static QuaternionD  rotMat2Quaternion(const RotationMatD& rotMat);
    static QuaternionF  rotMat2Quaternion(const RotationMatF& rotMat);

    static RotationMatD quaternion2RotMat(const QuaternionD& q);
    static RotationMatF quaternion2RotMat(const QuaternionF& q);

    static QuaternionD  rotVec2Quaternion(const RotationVecD& rotVec);
    static QuaternionF  rotVec2Quaternion(const RotationVecF& rotVec);

    static RotationVecD quaternion2RotVec(const QuaternionD& q);
    static RotationVecF quaternion2RotVec(const QuaternionF& q);

    static RotationMatD rotVec2RotMat(const RotationVecD& rotVec);
    static RotationMatF rotVec2RotMat(const RotationVecF& rotVec);

    static RotationVecD rotMat2RotVec(const RotationMatD& rotMat);
    static RotationVecF rotMat2RotVec(const RotationMatF& rotMat);

    static RotationVecD euler2RotVec(const EulerD& euler, const RotSequence& seq);
    static RotationVecF euler2RotVec(const EulerF& euler, const RotSequence& seq);

    static EulerD rotVec2Euler(const RotationVecD& rotVec, const RotSequence& seq);
    static EulerF rotVec2Euler(const RotationVecF& rotVec, const RotSequence& seq);

    static TranslationD rotatePos(const RotationMatD& rotMat, const TranslationD& trans);
    static TranslationF rotatePos(const RotationMatF& rotMat, const TranslationF& trans);

    static TranslationD transform(Matrix4x4D& tfMat, const TranslationD& trans);
    static TranslationF transform(Matrix4x4F& tfMat, const TranslationF& trans);

    static Matrix4x4D transform(const Matrix4x4D& tfMat, const Matrix4x4D& posture);
    static Matrix4x4F transform(const Matrix4x4F& tfMat, const Matrix4x4F& posture);

    static double clamp(const double &val,const double &min,const double &max);
    static float clamp(const float &val,const float &min,const float &max);
};

}

#endif 

