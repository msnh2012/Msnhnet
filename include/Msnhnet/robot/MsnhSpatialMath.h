#ifndef SPATIALMATH_H
#define SPATIALMATH_H

#include <Msnhnet/cv/MsnhCVMat.h>
#include <Msnhnet/cv/MsnhCVGeometry.h>

namespace Msnhnet
{

class MsnhNet_API SO3D:public RotationMatD
{
public:
    SO3D(){}

    SO3D(const Mat &mat); 

    SO3D(const SO3D& mat); 

    SO3D &operator= (Mat &mat);
    SO3D &operator= (SO3D &mat);

    RotationMatD &toRotMat();
    QuaternionD  toQuaternion();
    EulerD       toEuler(const RotSequence &rotSeq);
    RotationVecD toRotVector();

    SO3D adjoint();

    static SO3D rotX(double angleInRad);
    static SO3D rotY(double angleInRad);
    static SO3D rotZ(double angleInRad);
    static SO3D fromRotMat(const RotationMatD &rotMat);
    static SO3D fromQuaternion(const QuaternionD &quat);
    static SO3D fromEuler(const EulerD &euler, const RotSequence &rotSeq);
    static SO3D fromRotVec(const RotationVecD &rotVec);

    static Matrix3x3D wedge(const Vector3D &omg, bool needCalUnit=false);

    static Vector3D vee(const Matrix3x3D &mat3x3,bool needCalUnit=false);

    static SO3D exp(const Vector3D &omg);
    static SO3D exp(const Vector3D &omg, double theta);

    Vector3D log();

    static bool isSO3(const Mat &mat);

    static bool forceCheckSO3;
};

class MsnhNet_API SO3F:public RotationMatF
{
public:
    SO3F(){}

    SO3F(const Mat &mat); 

    SO3F(const SO3F& mat); 

    SO3F &operator= (Mat &mat);
    SO3F &operator= (SO3F &mat);

    RotationMatF &toRotMat();
    QuaternionF  toQuaternion();
    EulerF       toEuler(const RotSequence &rotSeq);
    RotationVecF toRotVector();

    SO3F adjoint();

    static SO3F rotX(float angleInRad);
    static SO3F rotY(float angleInRad);
    static SO3F rotZ(float angleInRad);
    static SO3F fromRotMat(const RotationMatF &rotMat);
    static SO3F fromQuaternion(const QuaternionF &quat);
    static SO3F fromEuler(const EulerF &euler, const RotSequence &rotSeq);
    static SO3F fromRotVec(const RotationVecF &rotVec);

    static Matrix3x3F wedge(const Vector3F &omg, bool needCalUnit=false);

    static Vector3F vee(const Matrix3x3F &mat3x3, bool needCalUnit=false);

    static SO3F exp(const Vector3F &omg);
    static SO3F exp(const Vector3F &omg,float theta);

    Vector3F log();

    static bool isSO3(const Mat &mat);

    static bool forceCheckSO3;
};

class MsnhNet_API SE3D:public Matrix4x4D
{
public:

    SE3D(){}
    SE3D(const Mat &mat); 

    SE3D(const SE3D& mat); 

    SE3D &operator= (Mat &mat);
    SE3D &operator= (SE3D &mat);

    Matrix4x4D &toMatrix4x4();

    Mat adjoint();

    static Matrix4x4D wedge(const ScrewD &screw, bool needCalUnit=false);

    static ScrewD vee(const Matrix4x4D &wed, bool needCalUnit=false);

    ScrewD log();

    static SE3D exp(const ScrewD &screw, double theta);

    static SE3D exp(const ScrewD &screw);

    static bool isSE3(const Mat &mat);

    static bool forceCheckSE3;
};

class MsnhNet_API SE3F:public Matrix4x4F
{
public:

    SE3F(){}
    SE3F(const Mat &mat); 

    SE3F(const SE3F& mat); 

    SE3F &operator= (Mat &mat);
    SE3F &operator= (SE3F &mat);

    Matrix4x4F &toMatrix4x4();

    Mat adjoint();

    static Matrix4x4F wedge(const ScrewF &screw, bool needCalUnit=false);

    static ScrewF vee(const Matrix4x4F &wed, bool needCalUnit=false);

    ScrewF log();

    static SE3F exp(const ScrewF &screw, float theta);

    static SE3F exp(const ScrewF &screw);

    static bool isSE3(const Mat &mat);

    static bool forceCheckSE3;
};

}

#endif 

