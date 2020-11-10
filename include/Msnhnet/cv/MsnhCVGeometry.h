#ifndef MSNHCVGEOMTRY_H
#define MSNHCVGEOMTRY_H

#include "Msnhnet/cv/MsnhCVMat.h"
#include "Msnhnet/cv/MsnhCVMatOp.h"

#ifndef M_PI
#define M_PI 3.14159265453
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

    static double deg2rad(double val);

    static double rad2deg(double val);

    static void singularityCheck(const int &group, double& theta);
};

}

#endif 

