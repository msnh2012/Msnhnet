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

    double a = euler.getVal(0);
    double b = euler.getVal(1);
    double c = euler.getVal(2);

    double sina = std::sin(a); 

    double sinb = std::sin(b); 

    double sinc = std::sin(c); 

    double cosa = std::cos(a); 

    double cosb = std::cos(b); 

    double cosc = std::cos(c); 

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
    default:
        return Rz*Ry*Rx;
    }
}

Quaternion Geometry::euler2Quaternion(Euler &euler, const RotSequence &seq)
{
    double a = euler.getVal(0)/2;
    double b = euler.getVal(1)/2;
    double c = euler.getVal(2)/2;

    double sina = std::sin(a); 

    double sinb = std::sin(b); 

    double sinc = std::sin(c); 

    double cosa = std::cos(a); 

    double cosb = std::cos(b); 

    double cosc = std::cos(c); 

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
    default:
        return Rz*Ry*Rx;
    }
}

Euler Geometry::rotMat2Euler(RotationMat &rotMat, const RotSequence &seq)
{
    const double m11 = rotMat.getValAtRowCol(0,0), m12 = rotMat.getValAtRowCol(0,1), m13 = rotMat.getValAtRowCol(0,2);
    const double m21 = rotMat.getValAtRowCol(1,0), m22 = rotMat.getValAtRowCol(1,1), m23 = rotMat.getValAtRowCol(1,2);
    const double m31 = rotMat.getValAtRowCol(2,0), m32 = rotMat.getValAtRowCol(2,1), m33 = rotMat.getValAtRowCol(2,2);

    double x = 0;
    double y = 0;
    double z = 0;

    switch (seq)
    {
    case ROT_XYZ:
        y = std::asin(clamp(m13,-1,1));
        if(std::abs(m13) < 0.999999)
        {
            x = std::atan2( - m23, m33 );
            z = std::atan2( - m12, m11 );
        }
        else
        {
            x = std::atan2( m32, m22 );
            z = 0;
        }
        break;
    case ROT_YXZ:
        x = std::asin( - clamp( m23, - 1, 1 ) );

        if ( std::abs( m23 ) < 0.999999)
        {
            y = std::atan2( m13, m33 );
            z = std::atan2( m21, m22 );
        }
        else
        {
            y = std::atan2( - m31, m11 );
            z = 0;
        }
        break;
    case ROT_ZXY:
        x = std::asin( clamp( m32, - 1, 1 ) );

        if ( std::abs( m32 ) < 0.999999 )
        {
            y = std::atan2( - m31, m33 );
            z = std::atan2( - m12, m22 );
        }
        else
        {
            y = 0;
            z = std::atan2( m21, m11 );
        }
        break;
    case ROT_ZYX:
        y = std::asin( - clamp( m31, - 1, 1 ) );

        if ( std::abs( m31 ) < 0.999999 )
        {
            x = std::atan2( m32, m33 );
            z = std::atan2( m21, m11 );
        }
        else
        {
            x = 0;
            z = std::atan2( - m12, m22 );
        }
        break;
    case ROT_YZX:
        z = std::asin( clamp( m21, - 1, 1 ) );

        if ( std::abs( m21 ) < 0.999999 )
        {
            x = std::atan2( - m23, m22 );
            y = std::atan2( - m31, m11 );
        }
        else
        {
            x = 0;
            y = std::atan2( m13, m33 );
        }
        break;
    case ROT_XZY:
        z = std::asin( - clamp( m12, - 1, 1 ) );

        if ( std::abs( m12 ) < 0.999999 )
        {
            x = std::atan2( m32, m22 );
            y = std::atan2( m13, m11 );
        }
        else
        {
            x = std::atan2( - m23, m33 );
            y = 0;
        }

        break;
    }

    Euler euler;
    euler.setVal({x,y,z});
    return euler;
}

Euler Geometry::quaternion2Euler(Quaternion &q, const RotSequence &seq)
{

    RotationMat rotMat = quaternion2RotMat(q);
    return rotMat2Euler(rotMat,seq);
}

Quaternion Geometry::rotMat2Quaternion(RotationMat &rotMat)
{
    double q[4];

    double trace = rotMat.getValAtRowCol(0,0)+rotMat.getValAtRowCol(1,1)+rotMat.getValAtRowCol(2,2);

    if(trace > ROT_EPS)

    {
        q[0] = 0.5*sqrt(1+rotMat.getValAtRowCol(0,0)+rotMat.getValAtRowCol(1,1)+rotMat.getValAtRowCol(2,2));
        q[1] = (rotMat.getValAtRowCol(2,1) - rotMat.getValAtRowCol(1,2))/(4*q[0]);
        q[2] = (rotMat.getValAtRowCol(0,2) - rotMat.getValAtRowCol(2,0))/(4*q[0]);
        q[3] = (rotMat.getValAtRowCol(1,0) - rotMat.getValAtRowCol(0,1))/(4*q[0]);
    }
    else if(rotMat.getValAtRowCol(0,0)>rotMat.getValAtRowCol(1,1) && rotMat.getValAtRowCol(0,0)>rotMat.getValAtRowCol(2,2))
    {
        double t = 2*sqrt(1+rotMat.getValAtRowCol(0,0)-rotMat.getValAtRowCol(1,1)-rotMat.getValAtRowCol(2,2));
        q[0]     = (rotMat.getValAtRowCol(2,1) - rotMat.getValAtRowCol(1,2))/t;
        q[1]     = 0.25*t;
        q[2]     = (rotMat.getValAtRowCol(0,2) + rotMat.getValAtRowCol(2,0))/t;
        q[3]     = (rotMat.getValAtRowCol(0,1) + rotMat.getValAtRowCol(1,0))/t;
    }
    else if(rotMat.getValAtRowCol(1,1)>rotMat.getValAtRowCol(0,0) && rotMat.getValAtRowCol(1,1)>rotMat.getValAtRowCol(2,2))
    {
        double t = 2*sqrt(1-rotMat.getValAtRowCol(0,0)+rotMat.getValAtRowCol(1,1)-rotMat.getValAtRowCol(2,2));
        q[0]     = (rotMat.getValAtRowCol(0,2) - rotMat.getValAtRowCol(2,0))/t;
        q[1]     = (rotMat.getValAtRowCol(0,1) + rotMat.getValAtRowCol(1,0))/t;
        q[2]     = 0.25*t;
        q[3]     = (rotMat.getValAtRowCol(2,1) + rotMat.getValAtRowCol(1,2))/t;
    }
    else if(rotMat.getValAtRowCol(2,2)>rotMat.getValAtRowCol(0,0) && rotMat.getValAtRowCol(2,2)>rotMat.getValAtRowCol(1,1))
    {
        double t = 2*sqrt(1-rotMat.getValAtRowCol(0,0)-rotMat.getValAtRowCol(1,1)+rotMat.getValAtRowCol(2,2));
        q[0]     = (rotMat.getValAtRowCol(1,0) - rotMat.getValAtRowCol(0,1))/t;
        q[1]     = (rotMat.getValAtRowCol(0,2) + rotMat.getValAtRowCol(2,0))/t;
        q[2]     = (rotMat.getValAtRowCol(1,2) + rotMat.getValAtRowCol(2,1))/t;
        q[3]     = 0.25*t;
    }

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
                      1.0-2*(q2*q2+q3*q3),  2.0*(q1*q2-q0*q3),         2.0*(q1*q3+q0*q2),
                      2.0*(q1*q2+q0*q3),        1.0-2*(q1*q1+q3*q3),   2.0*(q2*q3-q0*q1),
                      2.0*(q1*q3-q0*q2),        2.0*(q2*q3+q0*q1),         1.0-2*(q1*q1+q2*q2)
                  });

    return rotMat;
}

Quaternion Geometry::rotVec2Quaternion(RotationVec &rotVec)
{
    double Rx = rotVec.getFloat64()[0];
    double Ry = rotVec.getFloat64()[1];
    double Rz = rotVec.getFloat64()[2];

    double theta = sqrt(Rx*Rx+Ry*Ry+Rz*Rz);

    if(theta==0)
    {
        return Quaternion(1,0,0,0);
    }

    double kx = Rx/theta;
    double ky = Ry/theta;
    double kz = Rz/theta;

    double q0 = std::cos(0.5*theta);
    double q1 = kx*std::sin(0.5*theta);
    double q2 = ky*std::sin(0.5*theta);
    double q3 = kz*std::sin(0.5*theta);

    return Quaternion(q0,q1,q2,q3);
}

RotationVec Geometry::quaternion2RotVec(Quaternion &q)
{
    double theta = 2*acos(q.getQ0());
    RotationVec vec;

    if(theta==0)
    {
        vec.setVal({0,0,0});
        return vec;
    }

    double kx    = q.getQ1()/std::sin(0.5*theta);
    double ky    = q.getQ2()/std::sin(0.5*theta);
    double kz    = q.getQ3()/std::sin(0.5*theta);

    vec.setVal({kx*theta,ky*theta,kz*theta});
    return vec;
}

RotationMat Geometry::rotVec2RotMat(RotationVec &rotVec)
{
    Quaternion q = rotVec2Quaternion(rotVec);
    return quaternion2RotMat(q);
}

RotationVec Geometry::rotMat2RotVec(RotationMat &rotMat)
{
    Quaternion q = rotMat2Quaternion(rotMat);
    return quaternion2RotVec(q);
}

RotationVec Geometry::euler2RotVec(Euler &euler, const RotSequence &seq)
{
    Quaternion q = euler2Quaternion(euler,seq);
    return quaternion2RotVec(q);
}

Euler Geometry::rotVec2Euler(RotationVec & rotVec, const RotSequence &seq)
{
    Quaternion q = rotVec2Quaternion(rotVec);
    return quaternion2Euler(q,seq);
}

double Geometry::deg2rad(double val)
{
    return val/180*M_PI;
}

double Geometry::rad2deg(double val)
{
    return val*180/M_PI;
}

double Geometry::clamp(const double &val, const double &min, const double &max)
{
    if(val<min)
    {
        return min;
    }
    else if(val>max)
    {
        return max;
    }
    else
    {
        return val;
    }
}

}
