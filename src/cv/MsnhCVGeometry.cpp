#include "Msnhnet/cv/MsnhCVGeometry.h"

namespace Msnhnet
{

double deg2rad(const double &val)
{
    return val/180.0*M_PI;
}

float deg2rad(const float &val)
{
    return static_cast<float>(val/180.0*M_PI);
}

double rad2deg(const double &val)
{
    return val/M_PI*180.0;
}

float rad2deg(const float &val)
{
    return static_cast<float>(val/M_PI*180.0);
}

bool Geometry::isRealRotMat(Mat &R)
{
    Mat Rt = R.transpose();
    Mat shouldBeIdentity = Rt*R;
    Mat I = Mat::eye(3,shouldBeIdentity.getMatType());
    return  MatOp::norm(I, shouldBeIdentity) < 1e-5;
}

RotationMatD Geometry::euler2RotMat(const EulerD &euler, const RotSequence &seq)
{

    double a = euler[0];
    double b = euler[1];
    double c = euler[2];

    double sina = sin(a); 

    double sinb = sin(b); 

    double sinc = sin(c); 

    double cosa = cos(a); 

    double cosb = cos(b); 

    double cosc = cos(c); 

    RotationMatD Rx;
    Rx.setVal({
                  1 ,  0   ,   0   ,
                  0 , cosa , -sina ,
                  0 , sina ,  cosa
              });

    RotationMatD Ry;
    Ry.setVal({
                  cosb , 0 , sinb ,
                  0   , 1 ,   0  ,
                  -sinb , 0 , cosb
              });

    RotationMatD Rz;

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

RotationMatF Geometry::euler2RotMat(const EulerF &euler, const RotSequence &seq)
{

    float a = euler[0];
    float b = euler[1];
    float c = euler[2];

    float sina = sinf(a); 

    float sinb = sinf(b); 

    float sinc = sinf(c); 

    float cosa = cosf(a); 

    float cosb = cosf(b); 

    float cosc = cosf(c); 

    RotationMatF Rx;
    Rx.setVal({
                  1 ,  0   ,   0   ,
                  0 , cosa , -sina ,
                  0 , sina ,  cosa
              });

    RotationMatF Ry;
    Ry.setVal({
                  cosb , 0 , sinb ,
                  0   , 1 ,   0  ,
                  -sinb , 0 , cosb
              });

    RotationMatF Rz;

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

QuaternionD Geometry::euler2Quaternion(const EulerD &euler, const RotSequence &seq)
{
    double a = euler[0]/2.0;
    double b = euler[1]/2.0;
    double c = euler[2]/2.0;

    double sina = sin(a); 

    double sinb = sin(b); 

    double sinc = sin(c); 

    double cosa = cos(a); 

    double cosb = cos(b); 

    double cosc = cos(c); 

    QuaternionD Rx(cosa,sina,0,0);
    QuaternionD Ry(cosb,0,sinb,0);
    QuaternionD Rz(cosc,0,0,sinc);

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

QuaternionF Geometry::euler2Quaternion(const EulerF &euler, const RotSequence &seq)
{
    float a = euler[0]/2.0f;
    float b = euler[1]/2.0f;
    float c = euler[2]/2.0f;

    float sina = sinf(a); 

    float sinb = sinf(b); 

    float sinc = sinf(c); 

    float cosa = cosf(a); 

    float cosb = cosf(b); 

    float cosc = cosf(c); 

    QuaternionF Rx(cosa,sina,0,0);
    QuaternionF Ry(cosb,0,sinb,0);
    QuaternionF Rz(cosc,0,0,sinc);

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

EulerD Geometry::rotMat2Euler(const RotationMatD &rotMat, const RotSequence &seq)
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

    return EulerD({x,y,z});
}

EulerF Geometry::rotMat2Euler(const RotationMatF &rotMat, const RotSequence &seq)
{
    const float m11 = rotMat.getValAtRowCol(0,0), m12 = rotMat.getValAtRowCol(0,1), m13 = rotMat.getValAtRowCol(0,2);
    const float m21 = rotMat.getValAtRowCol(1,0), m22 = rotMat.getValAtRowCol(1,1), m23 = rotMat.getValAtRowCol(1,2);
    const float m31 = rotMat.getValAtRowCol(2,0), m32 = rotMat.getValAtRowCol(2,1), m33 = rotMat.getValAtRowCol(2,2);

    float x = 0;
    float y = 0;
    float z = 0;

    switch (seq)
    {
    case ROT_XYZ:
        y = asinf(clamp(m13,-1.f,1.f));
        if(fabsf(m13) < 0.999999f)
        {
            x = atan2f( - m23, m33 );
            z = atan2f( - m12, m11 );
        }
        else
        {
            x = atan2f( m32, m22 );
            z = 0;
        }
        break;
    case ROT_YXZ:
        x = asinf( - clamp( m23, -1.f, 1.f));

        if ( fabsf( m23 ) < 0.999999f)
        {
            y = atan2f( m13, m33 );
            z = atan2f( m21, m22 );
        }
        else
        {
            y = atan2f( - m31, m11 );
            z = 0;
        }
        break;
    case ROT_ZXY:
        x = asinf( clamp( m32, -1.f, 1.f) );

        if (fabsf( m32 ) < 0.999999f)
        {
            y = atan2f( - m31, m33 );
            z = atan2f( - m12, m22 );
        }
        else
        {
            y = 0;
            z = atan2f( m21, m11 );
        }
        break;
    case ROT_ZYX:
        y = asinf( - clamp( m31, -1.f, 1.f ));

        if ( fabsf( m31 ) < 0.999999f )
        {
            x = atan2f( m32, m33 );
            z = atan2f( m21, m11 );
        }
        else
        {
            x = 0;
            z = atan2f( - m12, m22 );
        }
        break;
    case ROT_YZX:
        z = asinf( clamp( m21, -1.f, 1.f));

        if ( fabsf( m21 ) < 0.999999f )
        {
            x = atan2f( - m23, m22 );
            y = atan2f( - m31, m11 );
        }
        else
        {
            x = 0;
            y = atan2f( m13, m33 );
        }
        break;
    case ROT_XZY:
        z = asinf( - clamp( m12, -1.f, 1.f));

        if ( fabsf( m12 ) < 0.999999f )
        {
            x = atan2f( m32, m22 );
            y = atan2f( m13, m11 );
        }
        else
        {
            x = atan2f( - m23, m33 );
            y = 0;
        }

        break;
    }

    return EulerF({x,y,z});
}

EulerD Geometry::quaternion2Euler(const QuaternionD &q, const RotSequence &seq)
{

    RotationMatD rotMat = quaternion2RotMat(q);
    return rotMat2Euler(rotMat,seq);
}

EulerF Geometry::quaternion2Euler(const QuaternionF &q, const RotSequence &seq)
{

    RotationMatF rotMat = quaternion2RotMat(q);
    return rotMat2Euler(rotMat,seq);
}

QuaternionD Geometry::rotMat2Quaternion(const RotationMatD &rotMat)
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

    return QuaternionD(q[0],q[1],q[2],q[3]);
}

QuaternionF Geometry::rotMat2Quaternion(const RotationMatF &rotMat)
{
    float q[4];

    float trace = rotMat.getValAtRowCol(0,0)+rotMat.getValAtRowCol(1,1)+rotMat.getValAtRowCol(2,2);

    if(trace > ROT_EPS)

    {
        q[0] = 0.5f*sqrtf(1+rotMat.getValAtRowCol(0,0)+rotMat.getValAtRowCol(1,1)+rotMat.getValAtRowCol(2,2));
        q[1] = (rotMat.getValAtRowCol(2,1) - rotMat.getValAtRowCol(1,2))/(4*q[0]);
        q[2] = (rotMat.getValAtRowCol(0,2) - rotMat.getValAtRowCol(2,0))/(4*q[0]);
        q[3] = (rotMat.getValAtRowCol(1,0) - rotMat.getValAtRowCol(0,1))/(4*q[0]);
    }
    else if(rotMat.getValAtRowCol(0,0)>rotMat.getValAtRowCol(1,1) && rotMat.getValAtRowCol(0,0)>rotMat.getValAtRowCol(2,2))
    {
        float t = 2*sqrtf(1+rotMat.getValAtRowCol(0,0)-rotMat.getValAtRowCol(1,1)-rotMat.getValAtRowCol(2,2));
        q[0]     = (rotMat.getValAtRowCol(2,1) - rotMat.getValAtRowCol(1,2))/t;
        q[1]     = 0.25f*t;
        q[2]     = (rotMat.getValAtRowCol(0,2) + rotMat.getValAtRowCol(2,0))/t;
        q[3]     = (rotMat.getValAtRowCol(0,1) + rotMat.getValAtRowCol(1,0))/t;
    }
    else if(rotMat.getValAtRowCol(1,1)>rotMat.getValAtRowCol(0,0) && rotMat.getValAtRowCol(1,1)>rotMat.getValAtRowCol(2,2))
    {
        float t = 2*sqrtf(1-rotMat.getValAtRowCol(0,0)+rotMat.getValAtRowCol(1,1)-rotMat.getValAtRowCol(2,2));
        q[0]     = (rotMat.getValAtRowCol(0,2) - rotMat.getValAtRowCol(2,0))/t;
        q[1]     = (rotMat.getValAtRowCol(0,1) + rotMat.getValAtRowCol(1,0))/t;
        q[2]     = 0.25f*t;
        q[3]     = (rotMat.getValAtRowCol(2,1) + rotMat.getValAtRowCol(1,2))/t;
    }
    else if(rotMat.getValAtRowCol(2,2)>rotMat.getValAtRowCol(0,0) && rotMat.getValAtRowCol(2,2)>rotMat.getValAtRowCol(1,1))
    {
        float t = 2*sqrtf(1-rotMat.getValAtRowCol(0,0)-rotMat.getValAtRowCol(1,1)+rotMat.getValAtRowCol(2,2));
        q[0]     = (rotMat.getValAtRowCol(1,0) - rotMat.getValAtRowCol(0,1))/t;
        q[1]     = (rotMat.getValAtRowCol(0,2) + rotMat.getValAtRowCol(2,0))/t;
        q[2]     = (rotMat.getValAtRowCol(1,2) + rotMat.getValAtRowCol(2,1))/t;
        q[3]     = 0.25f*t;
    }

    return QuaternionF(q[0],q[1],q[2],q[3]);
}

RotationMatD Geometry::quaternion2RotMat(const QuaternionD &q)
{
    double q0 = q.getQ0();
    double q1 = q.getQ1();
    double q2 = q.getQ2();
    double q3 = q.getQ3();

    RotationMatD rotMat;
    rotMat.setVal({
                      1.0-2*(q2*q2+q3*q3),  2.0*(q1*q2-q0*q3),     2.0*(q1*q3+q0*q2),
                      2.0*(q1*q2+q0*q3),    1.0-2*(q1*q1+q3*q3),   2.0*(q2*q3-q0*q1),
                      2.0*(q1*q3-q0*q2),    2.0*(q2*q3+q0*q1),     1.0-2*(q1*q1+q2*q2)
                  });

    return rotMat;
}

RotationMatF Geometry::quaternion2RotMat(const QuaternionF &q)
{
    float q0 = q.getQ0();
    float q1 = q.getQ1();
    float q2 = q.getQ2();
    float q3 = q.getQ3();

    RotationMatF rotMat;
    rotMat.setVal({
                      1.0f-2*(q2*q2+q3*q3),  2.0f*(q1*q2-q0*q3),     2.0f*(q1*q3+q0*q2),
                      2.0f*(q1*q2+q0*q3),    1.0f-2*(q1*q1+q3*q3),   2.0f*(q2*q3-q0*q1),
                      2.0f*(q1*q3-q0*q2),    2.0f*(q2*q3+q0*q1),     1.0f-2*(q1*q1+q2*q2)
                  });

    return rotMat;
}

QuaternionD Geometry::rotVec2Quaternion(const RotationVecD &rotVec)
{
    double Rx = rotVec[0];
    double Ry = rotVec[1];
    double Rz = rotVec[2];

    double theta = sqrt(Rx*Rx+Ry*Ry+Rz*Rz);

    if(theta==0)
    {
        return QuaternionD(1,0,0,0);
    }

    double kx = Rx/theta;
    double ky = Ry/theta;
    double kz = Rz/theta;

    double q0 = std::cos(0.5*theta);
    double q1 = kx*std::sin(0.5*theta);
    double q2 = ky*std::sin(0.5*theta);
    double q3 = kz*std::sin(0.5*theta);

    return QuaternionD(q0,q1,q2,q3);
}

QuaternionF Geometry::rotVec2Quaternion(const RotationVecF &rotVec)
{
    float Rx = rotVec[0];
    float Ry = rotVec[1];
    float Rz = rotVec[2];

    float theta = sqrtf(Rx*Rx+Ry*Ry+Rz*Rz);

    if(theta==0)
    {
        return QuaternionF(1,0,0,0);
    }

    float kx = Rx/theta;
    float ky = Ry/theta;
    float kz = Rz/theta;

    float q0 = cosf(0.5f*theta);
    float q1 = kx*sinf(0.5f*theta);
    float q2 = ky*sinf(0.5f*theta);
    float q3 = kz*sinf(0.5f*theta);

    return QuaternionF(q0,q1,q2,q3);
}

RotationVecD Geometry::quaternion2RotVec(const QuaternionD &q)
{
    double theta = 2*acos(q.getQ0());

    if(theta==0)
    {
        return RotationVecD({0,0,0});
    }

    double kx    = q.getQ1()/std::sin(0.5*theta);
    double ky    = q.getQ2()/std::sin(0.5*theta);
    double kz    = q.getQ3()/std::sin(0.5*theta);

    return RotationVecD({kx*theta,ky*theta,kz*theta});
}

RotationVecF Geometry::quaternion2RotVec(const QuaternionF &q)
{
    float theta = 2*acosf(q.getQ0());

    if(theta==0)
    {
        return RotationVecF({0,0,0});
    }

    float kx    = q.getQ1()/sinf(0.5f*theta);
    float ky    = q.getQ2()/sinf(0.5f*theta);
    float kz    = q.getQ3()/sinf(0.5f*theta);

    return RotationVecF({kx*theta,ky*theta,kz*theta});
}

RotationMatD Geometry::rotZ(double angle)
{

    double cosc  = cos(angle);
    double sinc  = sin(angle);

    RotationMatD Rz;

    Rz.setVal({
                  cosc , -sinc , 0 ,
                  sinc , cosc  , 0 ,
                  0   ,  0    , 1
              });
    return Rz;
}

RotationMatD Geometry::rotY(double angle)
{
    double cosb  = cos(angle);
    double sinb  = sin(angle);

    RotationMatD Ry;
    Ry.setVal({
                  cosb , 0 , sinb ,
                  0   , 1 ,   0  ,
                  -sinb , 0 , cosb
              });

    return Ry;
}

RotationMatD Geometry::rotX(double angle)
{
    double cosa  = cos(angle);
    double sina  = sin(angle);
    RotationMatD Rx;
    Rx.setVal({
                  1 ,  0   ,   0   ,
                  0 , cosa , -sina ,
                  0 , sina ,  cosa
              });
    return Rx;
}

RotationMatF Geometry::rotZ(float angle)
{

    float cosc  = static_cast<float>(cos(angle));
    float sinc  = static_cast<float>(sin(angle));

    RotationMatF Rz;

    Rz.setVal({
                  cosc , -sinc , 0 ,
                  sinc , cosc  , 0 ,
                  0   ,  0    , 1
              });
    return Rz;
}

RotationMatF Geometry::rotY(float angle)
{
    float cosb  = static_cast<float>(cos(angle));
    float sinb  = static_cast<float>(sin(angle));

    RotationMatF Ry;
    Ry.setVal({
                  cosb , 0 , sinb ,
                  0   , 1 ,   0  ,
                  -sinb , 0 , cosb
              });

    return Ry;

}

RotationMatF Geometry::rotX(float angle)
{
    float cosa  = static_cast<float>(cos(angle));
    float sina  = static_cast<float>(sin(angle));
    RotationMatF Rx;
    Rx.setVal({
                  1 ,  0   ,   0   ,
                  0 , cosa , -sina ,
                  0 , sina ,  cosa
              });
    return Rx;
}

RotationMatD Geometry::rotVec2RotMat(const RotationVecD &rotVec)
{
    QuaternionD q = rotVec2Quaternion(rotVec);
    return quaternion2RotMat(q);
}

RotationMatF Geometry::rotVec2RotMat(const RotationVecF &rotVec)
{
    QuaternionF q = rotVec2Quaternion(rotVec);
    return quaternion2RotMat(q);
}

RotationVecD Geometry::rotMat2RotVec(const RotationMatD &rotMat)
{
    QuaternionD q = rotMat2Quaternion(rotMat);
    return quaternion2RotVec(q);
}

RotationVecF Geometry::rotMat2RotVec(const RotationMatF &rotMat)
{
    QuaternionF q = rotMat2Quaternion(rotMat);
    return quaternion2RotVec(q);
}

RotationVecD Geometry::euler2RotVec(const EulerD &euler, const RotSequence &seq)
{
    QuaternionD q = euler2Quaternion(euler,seq);
    return quaternion2RotVec(q);
}

RotationVecF Geometry::euler2RotVec(const EulerF &euler, const RotSequence &seq)
{
    QuaternionF q = euler2Quaternion(euler,seq);
    return quaternion2RotVec(q);
}

EulerD Geometry::rotVec2Euler(const RotationVecD &rotVec, const RotSequence &seq)
{
    QuaternionD q = rotVec2Quaternion(rotVec);
    return quaternion2Euler(q,seq);
}

EulerF Geometry::rotVec2Euler(const RotationVecF &rotVec, const RotSequence &seq)
{
    QuaternionF q = rotVec2Quaternion(rotVec);
    return quaternion2Euler(q,seq);
}

TranslationD Geometry::rotatePos(const RotationMatD &rotMat, const TranslationD &trans)
{
    return TranslationD({rotMat.getValAtRowCol(0,0)*trans[0]+rotMat.getValAtRowCol(0,1)*trans[1]+rotMat.getValAtRowCol(0,2)*trans[2],
                         rotMat.getValAtRowCol(1,0)*trans[0]+rotMat.getValAtRowCol(1,1)*trans[1]+rotMat.getValAtRowCol(1,2)*trans[2],
                         rotMat.getValAtRowCol(2,0)*trans[0]+rotMat.getValAtRowCol(2,1)*trans[1]+rotMat.getValAtRowCol(2,2)*trans[2]});

}

TranslationF Geometry::rotatePos(const RotationMatF &rotMat, const TranslationF &trans)
{
    return     TranslationF({rotMat.getValAtRowCol(0,0)*trans[0]+rotMat.getValAtRowCol(0,1)*trans[1]+rotMat.getValAtRowCol(0,2)*trans[2],
                             rotMat.getValAtRowCol(1,0)*trans[0]+rotMat.getValAtRowCol(1,1)*trans[1]+rotMat.getValAtRowCol(1,2)*trans[2],
                             rotMat.getValAtRowCol(2,0)*trans[0]+rotMat.getValAtRowCol(2,1)*trans[1]+rotMat.getValAtRowCol(2,2)*trans[2]});
}

TranslationD Geometry::transform(Matrix4x4D &tfMat, const TranslationD &trans)
{
    return tfMat.mulVec3(trans);
}

TranslationF Geometry::transform(Matrix4x4F &tfMat, const TranslationF &trans)
{
    return tfMat.mulVec3(trans);
}

Matrix4x4D Geometry::transform(const Matrix4x4D &tfMat, const Matrix4x4D &posture)
{
    return tfMat*posture;
}

Matrix4x4F Geometry::transform(const Matrix4x4F &tfMat, const Matrix4x4F &posture)
{
    return tfMat*posture;
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

float Geometry::clamp(const float &val, const float &min, const float &max)
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

Matrix4x4D::Matrix4x4D():Mat_<4,4,double>()
{

}

Matrix4x4D::Matrix4x4D(const Mat &mat)
{
    if(mat.getWidth()!=4 || mat.getHeight()!=4 || mat.getChannel()!=1 || mat.getStep()!=8 || mat.getMatType()!= MatType::MAT_GRAY_F64)

    {
        throw Exception(1, "[Matrix4x4] mat should be: wxh==4x4 channel==1 step==8 matType==MAT_GRAY_F64", __FILE__, __LINE__,__FUNCTION__);
    }
    release();
    this->_channel  = mat.getChannel();
    this->_width    = mat.getWidth();
    this->_height   = mat.getHeight();
    this->_step     = mat.getStep();
    this->_matType  = mat.getMatType();

    if(mat.getBytes()!=nullptr)
    {
        uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
        memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
        this->_data.u8 =u8Ptr;
    }
}

Matrix4x4D::Matrix4x4D(const Matrix4x4D &mat)
{
    release();
    this->_channel  = mat.getChannel();
    this->_width    = mat.getWidth();
    this->_height   = mat.getHeight();
    this->_step     = mat.getStep();
    this->_matType  = mat.getMatType();

    if(mat.getBytes()!=nullptr)
    {
        uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
        memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
        this->_data.u8 =u8Ptr;
    }
}

Matrix4x4D::Matrix4x4D(const RotationMatD &rotMat)
{
    setRotationMat(rotMat);
}

Matrix4x4D::Matrix4x4D(const TranslationD &trans)
{
    setTranslation(trans);
}

Matrix4x4D::Matrix4x4D(const RotationMatD &rotMat, const TranslationD &trans):Mat_()
{
    this->setVal({
                     rotMat.getValAtRowCol(0,0), rotMat.getValAtRowCol(0,1), rotMat.getValAtRowCol(0,2), trans[0],
                     rotMat.getValAtRowCol(1,0), rotMat.getValAtRowCol(1,1), rotMat.getValAtRowCol(1,2), trans[1],
                     rotMat.getValAtRowCol(2,0), rotMat.getValAtRowCol(2,1), rotMat.getValAtRowCol(2,2), trans[2],
                     0,                          0,                          0,                          1
                 });
}

Matrix4x4D::Matrix4x4D(const std::vector<double> &val):Mat_<4,4,double>(val){}

Matrix4x4D &Matrix4x4D::operator=(Matrix4x4D &mat)
{
    if(this!=&mat)
    {
        release();
        this->_channel  = mat._channel;
        this->_width    = mat._width;
        this->_height   = mat._height;
        this->_step     = mat._step;
        this->_matType  = mat._matType;

        if(mat._data.u8!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat._data.u8, this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
    }
    return *this;
}

Matrix4x4D &Matrix4x4D::operator=(const Mat &mat)
{
    if(mat.getWidth()!=4 || mat.getHeight()!=4 || mat.getChannel()!=1 || mat.getStep()!=8 || mat.getMatType()!= MatType::MAT_GRAY_F64)

    {
        throw Exception(1, "[Matrix4x4] mat should be: wxh==4x4 channel==1 step==8 matType==MAT_GRAY_F64", __FILE__, __LINE__,__FUNCTION__);
    }

    if(this!=&mat)
    {
        release();
        this->_channel  = mat.getChannel();
        this->_width    = mat.getWidth();
        this->_height   = mat.getHeight();
        this->_step     = mat.getStep();
        this->_matType  = mat.getMatType();

        if(mat.getBytes()!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
    }
    return *this;
}

RotationMatD Matrix4x4D::getRotationMat() const
{
    RotationMatD rotMat;
    rotMat.setVal({
                      this->getValAtRowCol(0,0), this->getValAtRowCol(0,1),this->getValAtRowCol(0,2),
                      this->getValAtRowCol(1,0), this->getValAtRowCol(1,1),this->getValAtRowCol(1,2),
                      this->getValAtRowCol(2,0), this->getValAtRowCol(2,1),this->getValAtRowCol(2,2),
                  });
    return rotMat;
}

TranslationD Matrix4x4D::getTranslation() const
{
    return TranslationD({
                            this->getValAtRowCol(0,3),
                            this->getValAtRowCol(1,3),
                            this->getValAtRowCol(2,3)
                        });
}

void Matrix4x4D::setRotationMat(const RotationMatD &rotMat)
{
    this->setValAtRowCol(0,0,rotMat.getValAtRowCol(0,0));
    this->setValAtRowCol(0,1,rotMat.getValAtRowCol(0,1));
    this->setValAtRowCol(0,2,rotMat.getValAtRowCol(0,2));

    this->setValAtRowCol(1,0,rotMat.getValAtRowCol(1,0));
    this->setValAtRowCol(1,1,rotMat.getValAtRowCol(1,1));
    this->setValAtRowCol(1,2,rotMat.getValAtRowCol(1,2));

    this->setValAtRowCol(2,0,rotMat.getValAtRowCol(2,0));
    this->setValAtRowCol(2,1,rotMat.getValAtRowCol(2,1));
    this->setValAtRowCol(2,2,rotMat.getValAtRowCol(2,2));
}

void Matrix4x4D::setTranslation(const TranslationD &trans)
{
    this->setValAtRowCol(0,3,trans[0]);
    this->setValAtRowCol(1,3,trans[1]);
    this->setValAtRowCol(2,3,trans[2]);
}

void Matrix4x4D::translate(const Vector3D &vector)
{

    translate(vector[0], vector[1], vector[2]);
}

void Matrix4x4D::translate(const double &x, const double &y, const double &z)
{
    this->getFloat64()[3]  += x;
    this->getFloat64()[7]  += y;
    this->getFloat64()[11] += z;
}

void Matrix4x4D::rotate(const double &angle, const double &x, const double &y, const double &z)
{
    rotate(angle,Vector3D({x,y,z}));
}

void Matrix4x4D::rotate(const double &angle, const Vector3D &vector)
{
    Vector3D vec = vector;
    vec.normalize();
    double x = vec[0];
    double y = vec[1];
    double z = vec[2];

    RotationMatD rotMat = Geometry::euler2RotMat(EulerD({x*angle,y*angle,z*angle}),RotSequence::ROT_ZYX);
    this->setRotationMat(rotMat);
}

void Matrix4x4D::rotate(const EulerD &euler)
{
    RotationMatD rotMat = Geometry::euler2RotMat(euler,RotSequence::ROT_ZYX);
    this->setRotationMat(rotMat);
}

void Matrix4x4D::rotate(const QuaternionD &quat)
{
    RotationMatD rotMat = Geometry::quaternion2RotMat(quat);
    this->setRotationMat(rotMat);
}

void Matrix4x4D::scale(const double &x, const double &y, const double &z)
{
    this->getFloat64()[0]  *= x;
    this->getFloat64()[1]  *= x;
    this->getFloat64()[2]  *= x;

    this->getFloat64()[4]  *= y;
    this->getFloat64()[5]  *= y;
    this->getFloat64()[6]  *= y;

    this->getFloat64()[8]  *= z;
    this->getFloat64()[9]  *= z;
    this->getFloat64()[10] *= z;
}

void Matrix4x4D::scale(const Vector3D &vector)
{
    scale(vector[0],vector[1],vector[2]);
}

void Matrix4x4D::perspective(const double &verticalAngle, const double &aspectRatio, const double &nearPlane, const double &farPlane)
{

    if(nearPlane==farPlane || aspectRatio == 0.0)
    {
        return;
    }

    double radians = verticalAngle / 2.0;
    double sine     = sin(radians);
    if(sine == 0)
    {
        return;
    }

    double cotan = cos(radians) / sine;
    double clip  = farPlane - nearPlane;

    Matrix4x4D m = Matrix4x4D::eye();

    m.getFloat64()[0]  = cotan / aspectRatio;
    m.getFloat64()[5]  = cotan;
    m.getFloat64()[10] = -(nearPlane + farPlane) / clip;
    m.getFloat64()[11] = -(2.0 * nearPlane * farPlane) / clip;
    m.getFloat64()[14] = -1;
    m.getFloat64()[15] = 0;

    *this = *this * m;

}

void Matrix4x4D::ortho(const double &left, const double &right, const double &bottom, const double &top, const double &nearPlane, const double &farPlane)
{

    if (left == right || bottom == top || nearPlane == farPlane)
    {
        return;
    }

    double width     = right - left;
    double invheight = top - bottom;
    double clip      = farPlane - nearPlane;

    Matrix4x4D m = Matrix4x4D::eye();

    m.getFloat64()[0]  = 2.0 / width;
    m.getFloat64()[3]  = -(left + right) / width;
    m.getFloat64()[5]  = 2.0 / invheight;
    m.getFloat64()[7]  = -(top + bottom) / invheight;
    m.getFloat64()[10] = -2.0 / clip;
    m.getFloat64()[11] = -(nearPlane + farPlane) / clip;
    *this = *this * m;
}

void Matrix4x4D::lookAt(const Vector3D &eye, const Vector3D &center, const Vector3D &up)
{
    Vector3D forward = center - eye;

    if(forward.isFuzzyNull())
    {
        return;
    }

    forward.normalize();
    Vector3D side     = Vector3D::crossProduct(forward,up).normalized();
    Vector3D upVector = Vector3D::crossProduct(side, forward);

    this->getFloat64()[0]  = side[0];
    this->getFloat64()[1]  = side[1];
    this->getFloat64()[2]  = side[2];

    this->getFloat64()[4]  = upVector[0];
    this->getFloat64()[5]  = upVector[1];
    this->getFloat64()[6]  = upVector[2];

    this->getFloat64()[8]  = -forward[0];
    this->getFloat64()[9]  = -forward[1];
    this->getFloat64()[10] = -forward[2];

    this->getFloat64()[3]  = -Vector3D::dotProduct(side,eye);
    this->getFloat64()[7]  = -Vector3D::dotProduct(upVector,eye);
    this->getFloat64()[11] = Vector3D::dotProduct(forward,eye);
}

Vector3D Matrix4x4D::mulVec3(const Vector3D &vec3)
{
    double x = vec3[0]*this->getFloat64()[0] +
            vec3[1]*this->getFloat64()[1] +
            vec3[2]*this->getFloat64()[2] +
            this->getFloat64()[3];

    double y = vec3[0]*this->getFloat64()[4] +
            vec3[1]*this->getFloat64()[5] +
            vec3[2]*this->getFloat64()[6] +
            this->getFloat64()[7];

    double z = vec3[0]*this->getFloat64()[8] +
            vec3[1]*this->getFloat64()[9] +
            vec3[2]*this->getFloat64()[10] +
            this->getFloat64()[11];

    double w = vec3[0]*this->getFloat64()[12] +
            vec3[1]*this->getFloat64()[13] +
            vec3[2]*this->getFloat64()[14] +
            this->getFloat64()[15];

    if(w == 1.0)
    {
        return Vector3D({x,y,z});
    }
    else
    {
        return Vector3D({x/w,y/w,z/w});
    }
}

Matrix3x3D Matrix4x4D::normalMatrix()
{

    Matrix3x3D mat3x3 = Mat::eye(3,MatType::MAT_GRAY_F64);
    double det = this->det();
    if(abs(det)<MSNH_F64_EPS)
    {
        return mat3x3;
    }

    det = 1/det;

    mat3x3.setVal(0,(getValAtRowCol(1,1)*getValAtRowCol(2,2) - getValAtRowCol(2,1)*getValAtRowCol(1,2)) * det);

    mat3x3.setVal(1,(getValAtRowCol(1,0)*getValAtRowCol(2,2) - getValAtRowCol(1,2)*getValAtRowCol(2,0)) * det);

    mat3x3.setVal(2,(getValAtRowCol(1,0)*getValAtRowCol(2,1) - getValAtRowCol(1,1)*getValAtRowCol(2,0)) * det);

    mat3x3.setVal(3,(getValAtRowCol(0,1)*getValAtRowCol(2,2) - getValAtRowCol(2,1)*getValAtRowCol(0,2)) * det);

    mat3x3.setVal(4,(getValAtRowCol(0,0)*getValAtRowCol(2,2) - getValAtRowCol(0,2)*getValAtRowCol(2,0)) * det);

    mat3x3.setVal(5,(getValAtRowCol(0,0)*getValAtRowCol(2,1) - getValAtRowCol(0,1)*getValAtRowCol(2,0)) * det);;

    mat3x3.setVal(6,(getValAtRowCol(0,1)*getValAtRowCol(1,2) - getValAtRowCol(0,2)*getValAtRowCol(1,1)) * det);

    mat3x3.setVal(7,(getValAtRowCol(0,0)*getValAtRowCol(1,2) - getValAtRowCol(0,2)*getValAtRowCol(1,0)) * det);

    mat3x3.setVal(8,(getValAtRowCol(0,0)*getValAtRowCol(1,1) - getValAtRowCol(1,0)*getValAtRowCol(0,1)) * det);

    return mat3x3;
}

Matrix4x4F::Matrix4x4F():Mat_<4,4,float>()
{

}

Matrix4x4F::Matrix4x4F(const Mat &mat)
{
    if(mat.getWidth()!=4 || mat.getHeight()!=4 || mat.getChannel()!=1 || mat.getStep()!=4 || mat.getMatType()!= MatType::MAT_GRAY_F32) 

    {
        throw Exception(1, "[Matrix4x4F] mat should be: wxh==4x4 channel==1 step==4 matType==MAT_GRAY_F32", __FILE__, __LINE__,__FUNCTION__);
    }
    release();
    this->_channel  = mat.getChannel();
    this->_width    = mat.getWidth();
    this->_height   = mat.getHeight();
    this->_step     = mat.getStep();
    this->_matType  = mat.getMatType();

    if(mat.getBytes()!=nullptr)
    {
        uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
        memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
        this->_data.u8 =u8Ptr;
    }
}

Matrix4x4F::Matrix4x4F(const Matrix4x4F &mat)
{
    release();
    this->_channel  = mat.getChannel();
    this->_width    = mat.getWidth();
    this->_height   = mat.getHeight();
    this->_step     = mat.getStep();
    this->_matType  = mat.getMatType();

    if(mat.getBytes()!=nullptr)
    {
        uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
        memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
        this->_data.u8 =u8Ptr;
    }
}

Matrix4x4F::Matrix4x4F(const RotationMatF &rotMat, const TranslationF &trans)
{
    this->setVal({
                     rotMat.getValAtRowCol(0,0), rotMat.getValAtRowCol(0,1), rotMat.getValAtRowCol(0,2), trans[0],
                     rotMat.getValAtRowCol(1,0), rotMat.getValAtRowCol(1,1), rotMat.getValAtRowCol(1,2), trans[1],
                     rotMat.getValAtRowCol(2,0), rotMat.getValAtRowCol(2,1), rotMat.getValAtRowCol(2,2), trans[2],
                     0,                          0,                          0,                          1
                 });
}

Matrix4x4F::Matrix4x4F(const std::vector<float> &val):Mat_<4,4,float>(val){}

Matrix4x4F &Matrix4x4F::operator=(Matrix4x4F &mat)
{
    if(this!=&mat)
    {
        release();
        this->_channel  = mat._channel;
        this->_width    = mat._width;
        this->_height   = mat._height;
        this->_step     = mat._step;
        this->_matType  = mat._matType;

        if(mat._data.u8!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat._data.u8, this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
    }
    return *this;
}

Matrix4x4F &Matrix4x4F::operator=(const Mat &mat)
{
    if(mat.getWidth()!=4 || mat.getHeight()!=4 || mat.getChannel()!=1 || mat.getStep()!=4 || mat.getMatType()!= MatType::MAT_GRAY_F32)

    {
        throw Exception(1, "[Matrix4x4F] mat should be: wxh==4x4 channel==1 step==4 matType==MAT_GRAY_F32", __FILE__, __LINE__,__FUNCTION__);
    }

    if(this!=&mat)
    {
        release();
        this->_channel  = mat.getChannel();
        this->_width    = mat.getWidth();
        this->_height   = mat.getHeight();
        this->_step     = mat.getStep();
        this->_matType  = mat.getMatType();

        if(mat.getBytes()!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
    }
    return *this;
}

RotationMatF Matrix4x4F::getRotationMat() const
{
    RotationMatF rotMat;
    rotMat.setVal({
                      this->getValAtRowCol(0,0), this->getValAtRowCol(0,1),this->getValAtRowCol(0,2),
                      this->getValAtRowCol(1,0), this->getValAtRowCol(1,1),this->getValAtRowCol(1,2),
                      this->getValAtRowCol(2,0), this->getValAtRowCol(2,1),this->getValAtRowCol(2,2),
                  });
    return rotMat;
}

TranslationF Matrix4x4F::getTranslation() const
{
    return TranslationF({
                            this->getValAtRowCol(0,3),
                            this->getValAtRowCol(1,3),
                            this->getValAtRowCol(2,3)
                        });
}

void Matrix4x4F::setRotationMat(const RotationMatF &rotMat)
{
    this->setValAtRowCol(0,0,rotMat.getValAtRowCol(0,0));
    this->setValAtRowCol(0,1,rotMat.getValAtRowCol(0,1));
    this->setValAtRowCol(0,2,rotMat.getValAtRowCol(0,2));

    this->setValAtRowCol(1,0,rotMat.getValAtRowCol(1,0));
    this->setValAtRowCol(1,1,rotMat.getValAtRowCol(1,1));
    this->setValAtRowCol(1,2,rotMat.getValAtRowCol(1,2));

    this->setValAtRowCol(2,0,rotMat.getValAtRowCol(2,0));
    this->setValAtRowCol(2,1,rotMat.getValAtRowCol(2,1));
    this->setValAtRowCol(2,2,rotMat.getValAtRowCol(2,2));
}

void Matrix4x4F::setTranslation(const TranslationF &trans)
{
    this->setValAtRowCol(0,3,trans[0]);
    this->setValAtRowCol(1,3,trans[1]);
    this->setValAtRowCol(2,3,trans[2]);
}

void Matrix4x4F::translate(const Vector3F &vector)
{

    translate(vector[0], vector[1], vector[2]);
}

void Matrix4x4F::translate(const float &x, const float &y, const float &z)
{
    this->getFloat32()[3]  += x;
    this->getFloat32()[7]  += y;
    this->getFloat32()[11] += z;
}

void Matrix4x4F::rotate(const float &angle, const float &x, const float &y, const float &z)
{
    rotate(angle,Vector3F({x,y,z}));
}

void Matrix4x4F::rotate(const float &angle, const Vector3F &vector)
{
    Vector3F vec = vector;
    vec.normalize();
    float x = vec[0];
    float y = vec[1];
    float z = vec[2];

    RotationMatF rotMat = Geometry::euler2RotMat(EulerF({x*angle,y*angle,z*angle}),RotSequence::ROT_ZYX);
    this->setRotationMat(rotMat);
}

void Matrix4x4F::rotate(const EulerF &euler)
{
    RotationMatF rotMat = Geometry::euler2RotMat(euler,RotSequence::ROT_ZYX);
    this->setRotationMat(rotMat);
}

void Matrix4x4F::rotate(const QuaternionF &quat)
{
    RotationMatF rotMat = Geometry::quaternion2RotMat(quat);
    this->setRotationMat(rotMat);
}

void Matrix4x4F::scale(const float &x, const float &y, const float &z)
{
    this->getFloat32()[0]  *= x;
    this->getFloat32()[1]  *= x;
    this->getFloat32()[2]  *= x;

    this->getFloat32()[4]  *= y;
    this->getFloat32()[5]  *= y;
    this->getFloat32()[6]  *= y;

    this->getFloat32()[8]  *= z;
    this->getFloat32()[9]  *= z;
    this->getFloat32()[10] *= z;
}

void Matrix4x4F::scale(const Vector3F &vector)
{
    scale(vector[0],vector[1],vector[2]);
}

void Matrix4x4F::perspective(const float &verticalAngle, const float &aspectRatio, const float &nearPlane, const float &farPlane)
{

    if(nearPlane==farPlane || aspectRatio == 0.0f)
    {
        return;
    }

    float radians = verticalAngle / 2.0f;
    float sine    = sinf(radians);
    if(sine == 0)
    {
        return;
    }

    float cotan  = cosf(radians) / sine;
    float clip   = farPlane - nearPlane;

    Matrix4x4F m = Matrix4x4F::eye();

    m.getFloat32()[0]  = cotan / aspectRatio;
    m.getFloat32()[5]  = cotan;
    m.getFloat32()[10] = -(nearPlane + farPlane) / clip;
    m.getFloat32()[11] = -(2.0f * nearPlane * farPlane) / clip;
    m.getFloat32()[14] = -1.0f;
    m.getFloat32()[15] = 0;

    *this = *this * m;
}

void Matrix4x4F::ortho(const float &left, const float &right, const float &bottom, const float &top, const float &nearPlane, const float &farPlane)
{

    if (left == right || bottom == top || nearPlane == farPlane)
    {
        return;
    }

    float width     = right - left;
    float invheight = top - bottom;
    float clip      = farPlane - nearPlane;

    Matrix4x4F m = Matrix4x4F::eye();

    m.getFloat32()[0]  = 2.0f / width;
    m.getFloat32()[3]  = -(left + right) / width;
    m.getFloat32()[5]  = 2.0f / invheight;
    m.getFloat32()[7]  = -(top + bottom) / invheight;
    m.getFloat32()[10] = -2.0f / clip;
    m.getFloat32()[11] = -(nearPlane + farPlane) / clip;
    *this = *this * m;
}

void Matrix4x4F::lookAt(const Vector3F &eye, const Vector3F &center, const Vector3F &up)
{
    Vector3F forward = center - eye;

    if(forward.isFuzzyNull())
    {
        return;
    }

    forward.normalize();
    Vector3F side     = Vector3F::crossProduct(forward,up).normalized();
    Vector3F upVector = Vector3F::crossProduct(side, forward);

    this->getFloat32()[0]  = side[0];
    this->getFloat32()[1]  = side[1];
    this->getFloat32()[2]  = side[2];

    this->getFloat32()[4]  = upVector[0];
    this->getFloat32()[5]  = upVector[1];
    this->getFloat32()[6]  = upVector[2];

    this->getFloat32()[8]  = -forward[0];
    this->getFloat32()[9]  = -forward[1];
    this->getFloat32()[10] = -forward[2];

    this->getFloat32()[3]  = -Vector3F::dotProduct(side,eye);
    this->getFloat32()[7]  = -Vector3F::dotProduct(upVector,eye);
    this->getFloat32()[11] = Vector3F::dotProduct(forward,eye);

}

Vector3F Matrix4x4F::mulVec3(const Vector3F &vec3)
{
    float x = vec3[0]*this->getFloat32()[0] +
            vec3[1]*this->getFloat32()[1] +
            vec3[2]*this->getFloat32()[2] +
            this->getFloat32()[3];

    float y = vec3[0]*this->getFloat32()[4] +
            vec3[1]*this->getFloat32()[5] +
            vec3[2]*this->getFloat32()[6] +
            this->getFloat32()[7];

    float z = vec3[0]*this->getFloat32()[8] +
            vec3[1]*this->getFloat32()[9] +
            vec3[2]*this->getFloat32()[10] +
            this->getFloat32()[11];

    float w = vec3[0]*this->getFloat32()[12] +
            vec3[1]*this->getFloat32()[13] +
            vec3[2]*this->getFloat32()[14] +
            this->getFloat32()[15];

    if(w == 1.0f)
    {
        return Vector3F({x,y,z});
    }
    else
    {
        return Vector3F({x/w,y/w,z/w});
    }
}
Matrix3x3F Matrix4x4F::normalMatrix()
{

    Matrix3x3F mat3x3 = Mat::eye(3,MatType::MAT_GRAY_F32);
    float det = static_cast<float>(this->det());
    if(abs(det)<MSNH_F32_EPS)
    {
        return mat3x3;
    }

    det = 1.0f/det;

    mat3x3.setVal(0,(getValAtRowCol(1,1)*getValAtRowCol(2,2) - getValAtRowCol(2,1)*getValAtRowCol(1,2)) * det);

    mat3x3.setVal(1,(getValAtRowCol(1,0)*getValAtRowCol(2,2) - getValAtRowCol(1,2)*getValAtRowCol(2,0)) * det);

    mat3x3.setVal(2,(getValAtRowCol(1,0)*getValAtRowCol(2,1) - getValAtRowCol(1,1)*getValAtRowCol(2,0)) * det);

    mat3x3.setVal(3,(getValAtRowCol(0,1)*getValAtRowCol(2,2) - getValAtRowCol(2,1)*getValAtRowCol(0,2)) * det);

    mat3x3.setVal(4,(getValAtRowCol(0,0)*getValAtRowCol(2,2) - getValAtRowCol(0,2)*getValAtRowCol(2,0)) * det);

    mat3x3.setVal(5,(getValAtRowCol(0,0)*getValAtRowCol(2,1) - getValAtRowCol(0,1)*getValAtRowCol(2,0)) * det);;

    mat3x3.setVal(6,(getValAtRowCol(0,1)*getValAtRowCol(1,2) - getValAtRowCol(0,2)*getValAtRowCol(1,1)) * det);

    mat3x3.setVal(7,(getValAtRowCol(0,0)*getValAtRowCol(1,2) - getValAtRowCol(0,2)*getValAtRowCol(1,0)) * det);

    mat3x3.setVal(8,(getValAtRowCol(0,0)*getValAtRowCol(1,1) - getValAtRowCol(1,0)*getValAtRowCol(0,1)) * det);

    return mat3x3;
}

}
