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

class MsnhNet_API Geometry
{
public:
    static bool isRealRotMat(Mat &R);

    static RotationMat euler2RotMat(Euler& euler, const RotSequence& seq);

    static Quaternion  euler2Quaternion(Euler& euler, const RotSequence& seq);

    static Euler rotMat2Euler(RotationMat& rotMat, const RotSequence &seq);

    static Euler quaternion2Euler(Quaternion& q, const RotSequence& seq);

    static Quaternion  rotMat2Quaternion(RotationMat& rotMat);

    static RotationMat quaternion2RotMat(Quaternion& q);

    static Quaternion  rotVec2Quaternion(RotationVec& rotVec);

    static RotationVec quaternion2RotVec(Quaternion& q);

    static RotationMat rotVec2RotMat(RotationVec& rotVec);

    static RotationVec rotMat2RotVec(RotationMat& rotMat);

    static RotationVec euler2RotVec(Euler& euler, const RotSequence& seq);

    static Euler rotVec2Euler(RotationVec& rotVec, const RotSequence& seq);

    static double deg2rad(double val);

    static double rad2deg(double val);

    static double clamp(const double &val,const double &min,const double &max);
};

class Matrix4x4 : public Mat_<4,4,double>
{
public:
    Matrix4x4();

    Matrix4x4(const Mat &mat); 

    Matrix4x4(const Matrix4x4& mat); 

    Matrix4x4& operator= (Matrix4x4 &mat);

    Matrix4x4& operator= (const Mat &mat);
};

}

#endif 

