#include "Msnhnet/math/MsnhGeometryS.h"

namespace Msnhnet
{

RotationMatDS GeometryS::euler2RotMat(const EulerDS &euler, const RotSequence &seq)
{

    double a = euler.val[0];
    double b = euler.val[1];
    double c = euler.val[2];

    double sina = sin(a); 

    double sinb = sin(b); 

    double sinc = sin(c); 

    double cosa = cos(a); 

    double cosb = cos(b); 

    double cosc = cos(c); 

    /*
        RotMatDS Rx;
        Rx.setVal({
                      1 ,  0   ,   0   ,
                      0 , cosa , -sina ,
                      0 , sina ,  cosa
                  });

        RotMatDS Ry;
        Ry.setVal({
                      cosb , 0 , sinb ,
                      0   , 1 ,   0  ,
                      -sinb , 0 , cosb
                  });

        RotMatDS Rz;

        Rz.setVal({
                      cosc , -sinc , 0 ,
                      sinc , cosc  , 0 ,
                      0   ,  0    , 1
                  });
        */
    RotationMatDS R;

    switch (seq)
    {
    case ROT_XYZ:
        R.setVal({
                     cosb*cosc,                   -sinc*cosb,                 sinb,
                     sina*sinb*cosc + sinc*cosa, -sina*sinb*sinc + cosa*cosc, -sina*cosb,
                     sina*sinc - sinb*cosa*cosc, sina*cosc + sinb*sinc*cosa, cosa*cosb
                 });
        return R;
    case ROT_XZY:
        R.setVal({
                     cosb*cosc,                  -sinc,                       sinb*cosc,
                     sina*sinb + sinc*cosa*cosb, cosa*cosc, -sina*cosb + sinb*sinc*cosa,
                     sina*sinc*cosb - sinb*cosa, sina*cosc, sina*sinb*sinc + cosa*cosb
                 });
        return R;
    case ROT_YXZ:
        R.setVal({
                     sina*sinb*sinc + cosb*cosc, sina*sinb*cosc - sinc*cosb, sinb*cosa,
                     sinc*cosa, cosa*cosc, -sina,
                     sina*sinc*cosb - sinb*cosc, sina*cosb*cosc + sinb*sinc, cosa*cosb
                 });
        return R;
    case ROT_YZX:
        R.setVal({
                     cosb*cosc, sina*sinb - sinc*cosa*cosb, sina*sinc*cosb + sinb*cosa,
                     sinc,                     cosa*cosc,           -sina*cosc,
                     -sinb*cosc, sina*cosb + sinb*sinc*cosa, -sina*sinb*sinc + cosa*cosb
                 });
        return R;
    case ROT_ZXY:
        R.setVal({
                     -sina*sinb*sinc + cosb*cosc, -sinc*cosa, sina*sinc*cosb + sinb*cosc,
                     sina*sinb*cosc + sinc*cosb, cosa*cosc, -sina*cosb*cosc + sinb*sinc,
                     -sinb*cosa,                  sina,                cosa*cosb
                 });
        return R;
    case ROT_ZYX:
        R.setVal({
                     cosb*cosc, sina*sinb*cosc - sinc*cosa, sina*sinc + sinb*cosa*cosc,
                     sinc*cosb, sina*sinb*sinc + cosa*cosc, -sina*cosc + sinb*sinc*cosa,
                     -sinb,               sina*cosb,              cosa*cosb
                 });
        return R;
    default:

        R.setVal({
                     cosb*cosc, sina*sinb*cosc - sinc*cosa, sina*sinc + sinb*cosa*cosc,
                     sinc*cosb, sina*sinb*sinc + cosa*cosc, -sina*cosc + sinb*sinc*cosa,
                     -sinb,               sina*cosb,              cosa*cosb
                 });
        return R;
    }
}

RotationMatFS GeometryS::euler2RotMat(const EulerFS &euler, const RotSequence &seq)
{

    float a = euler.val[0];
    float b = euler.val[1];
    float c = euler.val[2];

    float sina = sinf(a); 

    float sinb = sinf(b); 

    float sinc = sinf(c); 

    float cosa = cosf(a); 

    float cosb = cosf(b); 

    float cosc = cosf(c); 

    /*
        RotMatFS Rx;
        Rx.setVal({
                      1 ,  0   ,   0   ,
                      0 , cosa , -sina ,
                      0 , sina ,  cosa
                  });

        RotMatFS Ry;
        Ry.setVal({
                      cosb , 0 , sinb ,
                      0   , 1 ,   0  ,
                      -sinb , 0 , cosb
                  });

        RotMatFS Rz;

        Rz.setVal({
                      cosc , -sinc , 0 ,
                      sinc , cosc  , 0 ,
                      0   ,  0    , 1
                  });
        */
    RotationMatFS R;

    switch (seq)
    {
    case ROT_XYZ:
        R.setVal({
                     cosb*cosc,                   -sinc*cosb,                 sinb,
                     sina*sinb*cosc + sinc*cosa, -sina*sinb*sinc + cosa*cosc, -sina*cosb,
                     sina*sinc - sinb*cosa*cosc, sina*cosc + sinb*sinc*cosa, cosa*cosb
                 });
        return R;
    case ROT_XZY:
        R.setVal({
                     cosb*cosc,                  -sinc,                       sinb*cosc,
                     sina*sinb + sinc*cosa*cosb, cosa*cosc, -sina*cosb + sinb*sinc*cosa,
                     sina*sinc*cosb - sinb*cosa, sina*cosc, sina*sinb*sinc + cosa*cosb
                 });
        return R;
    case ROT_YXZ:
        R.setVal({
                     sina*sinb*sinc + cosb*cosc, sina*sinb*cosc - sinc*cosb, sinb*cosa,
                     sinc*cosa, cosa*cosc, -sina,
                     sina*sinc*cosb - sinb*cosc, sina*cosb*cosc + sinb*sinc, cosa*cosb
                 });
        return R;
    case ROT_YZX:
        R.setVal({
                     cosb*cosc, sina*sinb - sinc*cosa*cosb, sina*sinc*cosb + sinb*cosa,
                     sinc,                     cosa*cosc,           -sina*cosc,
                     -sinb*cosc, sina*cosb + sinb*sinc*cosa, -sina*sinb*sinc + cosa*cosb
                 });
        return R;
    case ROT_ZXY:
        R.setVal({
                     -sina*sinb*sinc + cosb*cosc, -sinc*cosa, sina*sinc*cosb + sinb*cosc,
                     sina*sinb*cosc + sinc*cosb, cosa*cosc, -sina*cosb*cosc + sinb*sinc,
                     -sinb*cosa,                  sina,                cosa*cosb
                 });
        return R;
    case ROT_ZYX:
        R.setVal({
                     cosb*cosc, sina*sinb*cosc - sinc*cosa, sina*sinc + sinb*cosa*cosc,
                     sinc*cosb, sina*sinb*sinc + cosa*cosc, -sina*cosc + sinb*sinc*cosa,
                     -sinb,               sina*cosb,              cosa*cosb
                 });
        return R;
    default:

        R.setVal({
                     cosb*cosc, sina*sinb*cosc - sinc*cosa, sina*sinc + sinb*cosa*cosc,
                     sinc*cosb, sina*sinb*sinc + cosa*cosc, -sina*cosc + sinb*sinc*cosa,
                     -sinb,               sina*cosb,              cosa*cosb
                 });
        return R;
    }
}

QuaternionDS GeometryS::euler2Quaternion(const EulerDS &euler, const RotSequence &seq)
{
    double a = euler.val[0]/2.0;
    double b = euler.val[1]/2.0;
    double c = euler.val[2]/2.0;

    double sina = sin(a); 

    double sinb = sin(b); 

    double sinc = sin(c); 

    double cosa = cos(a); 

    double cosb = cos(b); 

    double cosc = cos(c); 

    QuaternionDS Rx(cosa,sina,0,0);
    QuaternionDS Ry(cosb,0,sinb,0);
    QuaternionDS Rz(cosc,0,0,sinc);

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

QuaternionFS GeometryS::euler2Quaternion(const EulerFS &euler, const RotSequence &seq)
{
    float a = euler.val[0]/2.0f;
    float b = euler.val[1]/2.0f;
    float c = euler.val[2]/2.0f;

    float sina = sinf(a); 

    float sinb = sinf(b); 

    float sinc = sinf(c); 

    float cosa = cosf(a); 

    float cosb = cosf(b); 

    float cosc = cosf(c); 

    QuaternionFS Rx(cosa,sina,0,0);
    QuaternionFS Ry(cosb,0,sinb,0);
    QuaternionFS Rz(cosc,0,0,sinc);

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

EulerDS GeometryS::rotMat2Euler(const RotationMatDS &rotMat, const RotSequence &seq)
{
    const double m11 = rotMat.val[0], m12 = rotMat.val[1], m13 = rotMat.val[2];
    const double m21 = rotMat.val[3], m22 = rotMat.val[4], m23 = rotMat.val[5];
    const double m31 = rotMat.val[6], m32 = rotMat.val[7], m33 = rotMat.val[8];

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

    return EulerDS(x,y,z);
}

EulerFS GeometryS::rotMat2Euler(const RotationMatFS &rotMat, const RotSequence &seq)
{
    const float m11 = rotMat.val[0], m12 = rotMat.val[1], m13 = rotMat.val[2];
    const float m21 = rotMat.val[3], m22 = rotMat.val[4], m23 = rotMat.val[5];
    const float m31 = rotMat.val[6], m32 = rotMat.val[7], m33 = rotMat.val[8];

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

    return EulerFS(x,y,z);
}

EulerDS GeometryS::quaternion2Euler(const QuaternionDS &q, const RotSequence &seq)
{

    RotationMatDS rotMat = quaternion2RotMat(q);
    return rotMat2Euler(rotMat,seq);
}

EulerFS GeometryS::quaternion2Euler(const QuaternionFS &q, const RotSequence &seq)
{

    RotationMatFS rotMat = quaternion2RotMat(q);
    return rotMat2Euler(rotMat,seq);
}

QuaternionDS GeometryS::rotMat2Quaternion(const RotationMatDS &rotMat)
{
    double q[4];

    double trace = rotMat(0,0)+rotMat(1,1)+rotMat(2,2);

    if(trace > ROT_EPS)
    {
        q[0] = 0.5*sqrt(1+rotMat(0,0)+rotMat(1,1)+rotMat(2,2));
        q[1] = (rotMat(2,1) - rotMat(1,2))/(4*q[0]);
        q[2] = (rotMat(0,2) - rotMat(2,0))/(4*q[0]);
        q[3] = (rotMat(1,0) - rotMat(0,1))/(4*q[0]);
    }
    else if(rotMat(0,0)>rotMat(1,1) && rotMat(0,0)>rotMat(2,2))
    {
        double t = 2*sqrt(1+rotMat(0,0)-rotMat(1,1)-rotMat(2,2));
        q[0]     = (rotMat(2,1) - rotMat(1,2))/t;
        q[1]     = 0.25*t;
        q[2]     = (rotMat(0,1) + rotMat(1,0))/t;
        q[3]     = (rotMat(0,2) + rotMat(2,0))/t;

    }
    else if(rotMat(1,1)>rotMat(0,0) && rotMat(1,1)>rotMat(2,2))
    {
        double t = 2*sqrt(1-rotMat(0,0)+rotMat(1,1)-rotMat(2,2));
        q[0]     = (rotMat(0,2) - rotMat(2,0))/t;
        q[1]     = (rotMat(0,1) + rotMat(1,0))/t;
        q[2]     = 0.25*t;
        q[3]     = (rotMat(2,1) + rotMat(1,2))/t;
    }
    else if(rotMat(2,2)>rotMat(0,0) && rotMat(2,2)>rotMat(1,1))
    {
        double t = 2*sqrt(1-rotMat(0,0)-rotMat(1,1)+rotMat(2,2));
        q[0]     = (rotMat(1,0) - rotMat(0,1))/t;
        q[1]     = (rotMat(0,2) + rotMat(2,0))/t;
        q[2]     = (rotMat(1,2) + rotMat(2,1))/t;
        q[3]     = 0.25*t;
    }

    return QuaternionDS(q[0],q[1],q[2],q[3]);
}

QuaternionFS GeometryS::rotMat2Quaternion(const RotationMatFS &rotMat)
{
    float q[4];

    float trace = rotMat(0,0)+rotMat(1,1)+rotMat(2,2);

    if(trace > ROT_EPS)
    {
        q[0] = 0.5f*sqrtf(1+rotMat(0,0)+rotMat(1,1)+rotMat(2,2));
        q[1] = (rotMat(2,1) - rotMat(1,2))/(4*q[0]);
        q[2] = (rotMat(0,2) - rotMat(2,0))/(4*q[0]);
        q[3] = (rotMat(1,0) - rotMat(0,1))/(4*q[0]);
    }
    else if(rotMat(0,0)>rotMat(1,1) && rotMat(0,0)>rotMat(2,2))
    {
        float t = 2*sqrtf(1+rotMat(0,0)-rotMat(1,1)-rotMat(2,2));
        q[0]     = (rotMat(2,1) - rotMat(1,2))/t;
        q[1]     = 0.25f*t;
        q[2]     = (rotMat(0,1) + rotMat(1,0))/t;
        q[3]     = (rotMat(0,2) + rotMat(2,0))/t;
    }
    else if(rotMat(1,1)>rotMat(0,0) && rotMat(1,1)>rotMat(2,2))
    {
        float t = 2*sqrtf(1-rotMat(0,0)+rotMat(1,1)-rotMat(2,2));
        q[0]     = (rotMat(0,2) - rotMat(2,0))/t;
        q[1]     = (rotMat(0,1) + rotMat(1,0))/t;
        q[2]     = 0.25f*t;
        q[3]     = (rotMat(2,1) + rotMat(1,2))/t;
    }
    else if(rotMat(2,2)>rotMat(0,0) && rotMat(2,2)>rotMat(1,1))
    {
        float t = 2*sqrtf(1-rotMat(0,0)-rotMat(1,1)+rotMat(2,2));
        q[0]     = (rotMat(1,0) - rotMat(0,1))/t;
        q[1]     = (rotMat(0,2) + rotMat(2,0))/t;
        q[2]     = (rotMat(1,2) + rotMat(2,1))/t;
        q[3]     = 0.25f*t;
    }

    return QuaternionFS(q[0],q[1],q[2],q[3]);
}

RotationMatDS GeometryS::quaternion2RotMat(const QuaternionDS &q)
{
    const double q0 = q.q0;
    const double q1 = q.q1;
    const double q2 = q.q2;
    const double q3 = q.q3;

    RotationMatDS rotMat;
    rotMat.setVal({
                      1.0-2*(q2*q2+q3*q3),  2.0*(q1*q2-q0*q3),     2.0*(q1*q3+q0*q2),
                      2.0*(q1*q2+q0*q3),    1.0-2*(q1*q1+q3*q3),   2.0*(q2*q3-q0*q1),
                      2.0*(q1*q3-q0*q2),    2.0*(q2*q3+q0*q1),     1.0-2*(q1*q1+q2*q2)
                  });

    return rotMat;
}

RotationMatFS GeometryS::quaternion2RotMat(const QuaternionFS &q)
{
    const float q0 = q.q0;
    const float q1 = q.q1;
    const float q2 = q.q2;
    const float q3 = q.q3;

    RotationMatFS rotMat;
    rotMat.setVal({
                      1.0f-2*(q2*q2+q3*q3),  2.0f*(q1*q2-q0*q3),     2.0f*(q1*q3+q0*q2),
                      2.0f*(q1*q2+q0*q3),    1.0f-2*(q1*q1+q3*q3),   2.0f*(q2*q3-q0*q1),
                      2.0f*(q1*q3-q0*q2),    2.0f*(q2*q3+q0*q1),     1.0f-2*(q1*q1+q2*q2)
                  });

    return rotMat;
}

QuaternionDS GeometryS::rotVec2Quaternion(const RotationVecDS &rotVec)
{
    const double Rx = rotVec[0];
    const double Ry = rotVec[1];
    const double Rz = rotVec[2];

    double theta = sqrt(Rx*Rx+Ry*Ry+Rz*Rz);

    if(theta==0)
    {
        return QuaternionDS(1,0,0,0);
    }

    const double kx = Rx/theta;
    const double ky = Ry/theta;
    const double kz = Rz/theta;

    const double sinTh = std::sin(0.5*theta);

    double q0 = std::cos(0.5*theta);
    double q1 = kx*sinTh;
    double q2 = ky*sinTh;
    double q3 = kz*sinTh;

    return QuaternionDS(q0,q1,q2,q3);
}

QuaternionFS GeometryS::rotVec2Quaternion(const RotationVecFS &rotVec)
{
    const float Rx = rotVec[0];
    const float Ry = rotVec[1];
    const float Rz = rotVec[2];

    float theta = sqrtf(Rx*Rx+Ry*Ry+Rz*Rz);

    if(theta==0)
    {
        return QuaternionFS(1,0,0,0);
    }

    const float kx = Rx/theta;
    const float ky = Ry/theta;
    const float kz = Rz/theta;

    const float sinTh = sinf(0.5f*theta);

    float q0 = cosf(0.5f*theta);
    float q1 = kx*sinTh;
    float q2 = ky*sinTh;
    float q3 = kz*sinTh;

    return QuaternionFS(q0,q1,q2,q3);
}

RotationVecDS GeometryS::quaternion2RotVec(const QuaternionDS &q)
{
    const double theta = 2*std::acos(q.q0);

    if(theta==0)
    {
        return RotationVecDS(0,0,0);
    }

    const double sinTh = std::sin(0.5*theta);

    const double kx    = q.q1/sinTh;
    const double ky    = q.q2/sinTh;
    const double kz    = q.q3/sinTh;

    return RotationVecDS(kx*theta,ky*theta,kz*theta);
}

RotationVecFS GeometryS::quaternion2RotVec(const QuaternionFS &q)
{
    const float theta = 2*acosf(q.q0);

    if(theta==0)
    {
        return RotationVecFS(0,0,0);
    }

    const float sinTh = sinf(0.5f*theta);

    const float kx    = q.q1/sinTh;
    const float ky    = q.q2/sinTh;
    const float kz    = q.q3/sinTh;

    return RotationVecFS(kx*theta,ky*theta,kz*theta);
}

RotationMatDS GeometryS::rotZ(double angleInRad)
{
    const double cosc  = cos(angleInRad);
    const double sinc  = sin(angleInRad);

    RotationMatDS Rz;

    Rz.setVal({
                  cosc , -sinc , 0 ,
                  sinc , cosc  , 0 ,
                  0   ,  0    , 1
              });
    return Rz;
}

RotationMatDS GeometryS::rotY(double angleInRad)
{
    const double cosb  = cos(angleInRad);
    const double sinb  = sin(angleInRad);

    RotationMatDS Ry;
    Ry.setVal({
                  cosb , 0 , sinb ,
                  0   , 1 ,   0  ,
                  -sinb , 0 , cosb
              });

    return Ry;
}

RotationMatDS GeometryS::rotX(double angleInRad)
{
    const double cosa  = cos(angleInRad);
    const double sina  = sin(angleInRad);
    RotationMatDS Rx;
    Rx.setVal({
                  1 ,  0   ,   0   ,
                  0 , cosa , -sina ,
                  0 , sina ,  cosa
              });
    return Rx;
}

RotationMatFS GeometryS::rotZ(float angleInRad)
{

    const float cosc  = cosf(angleInRad);
    const float sinc  = sinf(angleInRad);

    RotationMatFS Rz;

    Rz.setVal({
                  cosc , -sinc , 0 ,
                  sinc , cosc  , 0 ,
                  0   ,  0    , 1
              });
    return Rz;
}

RotationMatFS GeometryS::rotY(float angleInRad)
{
    const float cosb  = cosf(angleInRad);
    const float sinb  = sinf(angleInRad);

    RotationMatFS Ry;
    Ry.setVal({
                  cosb , 0 , sinb ,
                  0   , 1 ,   0  ,
                  -sinb , 0 , cosb
              });

    return Ry;
}

RotationMatFS GeometryS::rotX(float angleInRad)
{
    const float cosa  = cosf(angleInRad);
    const float sina  = sinf(angleInRad);
    RotationMatFS Rx;
    Rx.setVal({
                  1 ,  0   ,   0   ,
                  0 , cosa , -sina ,
                  0 , sina ,  cosa
              });
    return Rx;
}

RotationMatDS GeometryS::rotVec2RotMat(const RotationVecDS &rotVec)
{
    QuaternionDS q = rotVec2Quaternion(rotVec);
    return quaternion2RotMat(q);
}

RotationMatFS GeometryS::rotVec2RotMat(const RotationVecFS &rotVec)
{
    QuaternionFS q = rotVec2Quaternion(rotVec);
    return quaternion2RotMat(q);
}

RotationVecDS GeometryS::rotMat2RotVec(const RotationMatDS &rotMat)
{
    QuaternionDS q = rotMat2Quaternion(rotMat);
    return quaternion2RotVec(q);
}

RotationVecFS GeometryS::rotMat2RotVec(const RotationMatFS &rotMat)
{
    QuaternionFS q = rotMat2Quaternion(rotMat);
    return quaternion2RotVec(q);
}

RotationVecDS GeometryS::euler2RotVec(const EulerDS &euler, const RotSequence &seq)
{
    QuaternionDS q = euler2Quaternion(euler,seq);
    return quaternion2RotVec(q);
}

RotationVecFS GeometryS::euler2RotVec(const EulerFS &euler, const RotSequence &seq)
{
    QuaternionFS q = euler2Quaternion(euler,seq);
    return quaternion2RotVec(q);
}

EulerDS GeometryS::rotVec2Euler(const RotationVecDS &rotVec, const RotSequence &seq)
{
    QuaternionDS q = rotVec2Quaternion(rotVec);
    return quaternion2Euler(q,seq);
}

EulerFS GeometryS::rotVec2Euler(const RotationVecFS &rotVec, const RotSequence &seq)
{
    QuaternionFS q = rotVec2Quaternion(rotVec);
    return quaternion2Euler(q,seq);
}

TranslationDS GeometryS::rotatePos(const RotationMatDS &rotMat, const TranslationDS &trans)
{
    return TranslationDS(rotMat(0,0)*trans[0]+rotMat(0,1)*trans[1]+rotMat(0,2)*trans[2],
            rotMat(1,0)*trans[0]+rotMat(1,1)*trans[1]+rotMat(1,2)*trans[2],
            rotMat(2,0)*trans[0]+rotMat(2,1)*trans[1]+rotMat(2,2)*trans[2]);

}

TranslationFS GeometryS::rotatePos(const RotationMatFS &rotMat, const TranslationFS &trans)
{
    return TranslationFS(rotMat(0,0)*trans[0]+rotMat(0,1)*trans[1]+rotMat(0,2)*trans[2],
            rotMat(1,0)*trans[0]+rotMat(1,1)*trans[1]+rotMat(1,2)*trans[2],
            rotMat(2,0)*trans[0]+rotMat(2,1)*trans[1]+rotMat(2,2)*trans[2]);
}

}

