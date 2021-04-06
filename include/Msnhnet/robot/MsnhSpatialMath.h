#ifndef SPATIALMATH_H
#define SPATIALMATH_H

#include <Msnhnet/cv/MsnhCVMat.h>
#include <Msnhnet/cv/MsnhCVGeometry.h>

namespace Msnhnet
{

class SO3D:public RotationMatD
{
public:

    SO3D(const Mat &mat); 

    SO3D(const SO3D& mat); 

    SO3D &operator= (Mat &mat);
    SO3D &operator= (SO3D &mat);

    RotationMatD &toRotMat();
    QuaternionD  toQuaternion();
    EulerD       toEuler(const RotSequence &rotSeq);
    RotationVecD toRotVector();

    SO3D adjoint();

    static SO3D rotX(float angleInRad);
    static SO3D rotY(float angleInRad);
    static SO3D rotZ(float angleInRad);
    static SO3D fromRotMat(const RotationMatD &rotMat);
    static SO3D fromQuaternion(const QuaternionD &quat);
    static SO3D fromEuler(const EulerD &euler, const RotSequence &rotSeq);
    static SO3D fromRotVec(const RotationVecD &rotVec);

    static Matrix3x3D wedge(const Vector3D &vec3);
    static Vector3D vee(const Matrix3x3D &mat3x3);

    static SO3D exp(const Vector3D &vec3);
    static SO3D exp(const Vector3D &vec3,double theta);

    Vector3D log();

    static bool isSO3(Mat mat);

    bool forceCheckSO3 = true;
};

class SO3F:public RotationMatF
{

};

}

#endif 

