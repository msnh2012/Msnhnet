#include "Msnhnet/cv/MsnhCVGeometry.h"

namespace Msnhnet
{

bool Geometry::isRealRotMat(Mat &R)
{
    Mat Rt = R.transpose();
    Mat shouldBeIdentity = Rt*R;
    Mat I = Mat::eye(3,shouldBeIdentity.getMatType());
    return  MatOp::norm(I, shouldBeIdentity) < 1e-5;
}

RotationMat Geometry::euler2RotMat(Euler &euler, const RotSequence &seq)
{

    double a = deg2rad(euler.getVal(0));
    double b = deg2rad(euler.getVal(1));
    double c = deg2rad(euler.getVal(2));

    double sina = sin(a); 

    double sinb = sin(b); 

    double sinc = sin(c); 

    double cosa = cos(a); 

    double cosb = cos(b); 

    double cosc = cos(c); 

    RotationMat Rx;
    Rx.setVal({
                  1 ,  0   ,   0   ,
                  0 , cosa , -sina ,
                  0 , sina ,  cosa
              });

    RotationMat Ry;
    Ry.setVal({
                  cosb , 0 , sinb ,
                  0   , 1 ,   0  ,
                  -sinb , 0 , cosb
              });

    RotationMat Rz;

    Rz.setVal({
                  cosc , -sinc , 0 ,
                  sinc , cosc  , 0 ,
                  0   ,  0    , 1
              });

    /*TODO: exact all*/
    switch (seq)
    {
    case ROT_XYZ:
        return Rx*Ry*Rz;
    case ROT_XZY:
        return Rx*Rz*Ry;
    case ROT_YXZ:
        return Ry*Rx*Rz;
    case ROT_YZX:
        return Ry*Rz*Rx;
    case ROT_ZXY:
        return Rz*Rx*Ry;
    case ROT_ZYX:
        return Rz*Ry*Rx;
    case ROT_YXY:
        return Ry*Rx*Ry;
    case ROT_ZXZ:
        return Rz*Ry*Rz;
    case ROT_XYX:
        return Rx*Ry*Rx;
    case ROT_ZYZ:
        return Rz*Ry*Rz;
    case ROT_XZX:
        return Rx*Rz*Rx;
    case ROT_YZY:
        return Ry*Rz*Ry;
    default:
        return Rz*Ry*Rx;
    }
}

Quaternion Geometry::euler2Quaternion(Euler &euler, const RotSequence &seq)
{
    double a = deg2rad(euler.getVal(0))/2;
    double b = deg2rad(euler.getVal(1))/2;
    double c = deg2rad(euler.getVal(2))/2;

    double sina = sin(a); 

    double sinb = sin(b); 

    double sinc = sin(c); 

    double cosa = cos(a); 

    double cosb = cos(b); 

    double cosc = cos(c); 

    Quaternion Rx(cosa,sina,0,0);
    Quaternion Ry(cosb,0,sinb,0);
    Quaternion Rz(cosc,0,0,sinc);

    /*TODO: exact all*/
    switch (seq)
    {
    case ROT_XYZ:
        return Rx*Ry*Rz;
    case ROT_XZY:
        return Rx*Rz*Ry;
    case ROT_YXZ:
        return Ry*Rx*Rz;
    case ROT_YZX:
        return Ry*Rz*Rx;
    case ROT_ZXY:
        return Rz*Rx*Ry;
    case ROT_ZYX:
        return Rz*Ry*Rx;
    case ROT_YXY:
        return Ry*Rx*Ry;
    case ROT_ZXZ:
        return Rz*Ry*Rz;
    case ROT_XYX:
        return Rx*Ry*Rx;
    case ROT_ZYZ:
        return Rz*Ry*Rz;
    case ROT_XZX:
        return Rx*Rz*Rx;
    case ROT_YZY:
        return Ry*Rz*Ry;
    default:
        return Rz*Ry*Rx;
    }
}

Euler Geometry::rotMat2Euler(RotationMat &rotMat, const RotSequence &seq)
{
    Quaternion quat = rotMat2Quaternion(rotMat);
    return quaternion2Euler(quat,seq);
}

Euler Geometry::quaternion2Euler(Quaternion &q, const RotSequence &seq)
{
    double psi = 0;
    double theta = 0;
    double phi = 0;

    switch (seq)
    {
    case ROT_XYZ:
        psi   = atan2(2 * (q[1] * q[0] - q[2] * q[3]),(q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]));
        theta = asin(2 * (q[1] * q[3] + q[2] * q[0]));
        phi   = atan2(2 * (q[3] * q[0] - q[1] * q[2]), (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]));
        singularityCheck(2,theta);
        break;
    case ROT_XZY:
        psi   = atan2(2 * (q[1] * q[0] + q[2] * q[3]), (q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]));
        theta = asin(2 * (q[3] * q[0] - q[1] * q[2]));
        phi   = atan2(2 * (q[1] * q[3] + q[2] * q[0]), (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]));
        singularityCheck(2, theta);
        break;
    case ROT_YXZ:
        psi   = atan2(2 * (q[1] * q[3] + q[2] * q[0]), (q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]));
        theta = asin(2 * (q[1] * q[0] - q[2] * q[3]));
        phi   = atan2(2 * (q[1] * q[2] + q[3] * q[0]), (q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]));
        singularityCheck(2, theta);
        break;
    case ROT_YZX:
        psi   = atan2(2 * (q[2] * q[0] - q[1] * q[3]), (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]));
        theta = asin(2 * (q[1] * q[2] + q[3] * q[0]));
        phi   = atan2(2 * (q[1] * q[0] - q[3] * q[2]), (q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]));
        singularityCheck(2, theta);
        break;
    case ROT_ZXY:
        psi   = atan2(2 * (q[3] * q[0] - q[1] * q[2]), (q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]));
        theta = asin(2 * (q[1] * q[0] + q[2] * q[3]));
        phi   = atan2(2 * (q[2] * q[0] - q[3] * q[1]), (q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]));
        singularityCheck(2, theta);
        break;
    case ROT_ZYX:
        psi   = atan2(2 * (q[1] * q[2] + q[3] * q[0]), (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]));
        theta = asin(2 * (q[2] * q[0] - q[1] * q[3]));
        phi   = atan2(2 * (q[1] * q[0] + q[3] * q[2]), (q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]));
        singularityCheck(2, theta);
        break;
    case ROT_YXY:
        psi = atan2((q[1] * q[2] - q[3] * q[0]), (q[1] * q[0] + q[2] * q[3]));
        theta = acos(q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]);
        phi = atan2((q[1] * q[2] + q[3] * q[0]), (q[1] * q[0] - q[2] * q[3]));
        singularityCheck(1, theta);
        break;
    case ROT_ZXZ:
        psi = atan2((q[1] * q[3] + q[2] * q[0]), (q[1] * q[0] - q[2] * q[3]));
        theta = acos(q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]);
        phi = atan2((q[1] * q[3] - q[2] * q[0]), (q[1] * q[0] + q[2] * q[3]));
        singularityCheck(1, theta);
        break;
    case ROT_XYX:
        psi = atan2((q[1] * q[2] + q[3] * q[0]), (q[2] * q[0] - q[1] * q[3]));
        theta = acos(q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]);
        phi = atan2((q[1] * q[2] - q[3] * q[0]), (q[1] * q[3] + q[2] * q[0]));
        singularityCheck(1, theta);
        break;
    case ROT_ZYZ:
        psi = atan2((q[2] * q[3] - q[1] * q[0]), (q[1] * q[3] + q[2] * q[0]));
        theta = acos(q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]);
        phi = atan2((q[1] * q[0] + q[2] * q[3]), (q[2] * q[0] - q[1] * q[3]));
        singularityCheck(1, theta);
        break;
    case ROT_XZX:
        psi = atan2((q[1] * q[3] - q[2] * q[0]), (q[1] * q[2] + q[3] * q[0]));
        theta = acos(q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]);
        phi = atan2((q[1] * q[3] + q[2] * q[0]), (q[3] * q[0] - q[1] * q[2]));
        singularityCheck(1, theta);
        break;
    case ROT_YZY:
        psi = atan2((q[1] * q[0] + q[2] * q[3]), (q[3] * q[0] - q[1] * q[2]));
        theta = acos(q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]);
        phi = atan2((q[2] * q[3] - q[1] * q[0]), (q[1] * q[2] + q[3] * q[0]));
        singularityCheck(1, theta);
        break;
    default:
        psi   = atan2(2 * (q[1] * q[2] + q[3] * q[0]), (q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]));
        theta = asin(2 * (q[2] * q[0] - q[1] * q[3]));
        phi   = atan2(2 * (q[1] * q[0] + q[3] * q[2]), (q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]));
        singularityCheck(2, theta);
        break;
    }

    Euler euler;
    euler.setVal({rad2deg(psi),rad2deg(theta),rad2deg(phi)});
    return euler;
}

Quaternion Geometry::rotMat2Quaternion(RotationMat &rotMat)
{
    if(isRealRotMat(rotMat))
    {
        throw Exception(1, "[Geometry]: is not a rotation matrix! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    double q[4];
    q[0] = 0.5*sqrt(1+rotMat.getVal(0,0)+rotMat.getVal(1,1)+rotMat.getVal(2,2));
    q[1] = (rotMat.getVal(2,1) - rotMat.getVal(1,2))/(4*q[0]);
    q[2] = (rotMat.getVal(0,2) - rotMat.getVal(2,0))/(4*q[0]);
    q[3] = (rotMat.getVal(1,0) - rotMat.getVal(0,1))/(4*q[0]);

    return Quaternion(q[0],q[1],q[2],q[3]);
}

RotationMat Geometry::quaternion2RotMat(Quaternion &q)
{
    double q0 = q.getQ0();
    double q1 = q.getQ1();
    double q2 = q.getQ2();
    double q3 = q.getQ3();

    RotationMat rotMat;
    rotMat.setVal({
                      q0*q0+q1*q1-q2*q2-q3*q3,  2.0*(q1*q2-q0*q3),         2.0*(q1*q3+q0*q2),
                      2.0*(q1*q2+q0*q3),        q0*q0-q1*q1+q2*q2-q3*q3,   2.0*(q2*q3-q0*q1),
                      2.0*(q1*q3-q0*q2),        2.0*(q2*q3+q0*q1),         q0*q0-q1*q1-q2*q2+q3*q3
                  });

    if(isRealRotMat(rotMat))
    {
        throw Exception(1, "[Geometry]: is not a rotation matrix! \n",__FILE__,__LINE__,__FUNCTION__);
    }

    return rotMat;
}

double Geometry::deg2rad(double val)
{
    return val/180*M_PI;
}

double Geometry::rad2deg(double val)
{
    return val*180/M_PI;
}

void Geometry::singularityCheck(const int &group, double &theta)
{
    if (group == 1 && (M_PI - theta < M_PI / 180 || theta < M_PI / 180))
    {

        throw Exception(1, "[Geometry]: singularity check failed \n",__FILE__,__LINE__,__FUNCTION__);
    }
    else if (group == 2 && fabs(theta - M_PI / 2) < M_PI / 180)
    {

        throw Exception(1, "[Geometry]: singularity check failed \n",__FILE__,__LINE__,__FUNCTION__);
    }
    else if (group != 1 && group != 2)
    {
        throw Exception(1, "[Geometry]: group not 1 or 2 \n",__FILE__,__LINE__,__FUNCTION__);
    }
}

}
