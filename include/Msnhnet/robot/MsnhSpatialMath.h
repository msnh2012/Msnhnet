#ifndef SPATIALMATH_H
#define SPATIALMATH_H

#include <Msnhnet/cv/MsnhCVMat.h>
#include <Msnhnet/cv/MsnhCVGeometry.h>

namespace Msnhnet
{

template<typename T>
class Screw
{
public:
    Screw(){}
    Screw(Vector<3,T> v, Vector<3,T> w)
    {
        this->v = v;
        this->w = w;
    }

    Screw(Vector<6,T> list)
    {
        fromVector(list);
    }

    Vector<3,T> v;
    Vector<3,T> w;

    void print()
    {
        v.print();
        w.print();
    }

    friend Screw operator *(const Screw& screw, T a)
    {
        Screw tmp = screw;
        tmp.v = tmp.v*a;
        tmp.w = tmp.w*a;
        return tmp;
    }

    friend Screw operator *( T a, const Screw& screw)
    {
        Screw tmp = screw;
        tmp.v = tmp.v*a;
        tmp.w = tmp.w*a;
        return tmp;
    }

    Mat toMat()
    {
        Mat_<1,6,T> tmp;
        tmp.setVal({v[0],v[1],v[2],w[0],w[1],w[2]});
        return tmp;
    }

    Vector<6,T> toVector()
    {
        return Vector<6,double>({v[0],v[1],v[2],
                                 w[0],w[1],w[2]});
    }

    void fromVector(const Vector<6,T>& list)
    {
        v[0] = list[0];
        v[1] = list[1];
        v[2] = list[2];

        w[0] = list[3];
        w[1] = list[4];
        w[2] = list[5];
    }
};

typedef Screw<float> ScrewF;
typedef Screw<double> ScrewD;

class MsnhNet_API SO3D:public RotationMatD
{
public:
    SO3D(){}

    SO3D(const Mat &mat); 

    SO3D(Mat&& mat);
    SO3D(const SO3D& mat); 

    SO3D(SO3D&& mat);
    SO3D &operator= (const Mat &mat);
    SO3D &operator= (Mat&& mat);
    SO3D &operator= (const SO3D &mat);
    SO3D &operator= (SO3D&& mat);

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

    SO3D fastInvert();

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

    SO3F(Mat&& mat);
    SO3F(const SO3F& mat); 

    SO3F(SO3F&& mat);
    SO3F &operator= (const Mat &mat);
    SO3F &operator= (Mat&& mat);
    SO3F &operator= (const SO3F &mat);
    SO3F &operator= (SO3F&& mat);

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

    SO3F fastInvert();

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

    SE3D(Mat&& mat);
    SE3D(const SE3D& mat); 

    SE3D(SE3D&& mat);
    SE3D &operator= (const Mat &mat);
    SE3D &operator= (Mat&& mat);
    SE3D &operator= (const SE3D &mat);
    SE3D &operator= (SE3D&& mat);

    SE3D(const SO3D &rotMat, const Vector3D &trans);

    Matrix4x4D &toMatrix4x4();

    Mat adjoint();

    SE3D fastInvert();

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
    SE3F(Mat&& mat);
    SE3F(const SE3F& mat);
    SE3F(SE3F&& mat);
    SE3F &operator= (const Mat &mat);
    SE3F &operator= (Mat&& mat);
    SE3F &operator= (const SE3F &mat);
    SE3F &operator= (SE3F&& mat);

    SE3F(const SO3F &rotMat, const Vector3F &trans);

    Matrix4x4F &toMatrix4x4();

    Mat adjoint();

    SE3F fastInvert();

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

