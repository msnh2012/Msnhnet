#include "Msnhnet/robot/MsnhFrame.h"

namespace Msnhnet
{

void Twist::print()
{
    std::cout<<"================== Twist ==================="<<std::endl;
    v.print();
    omg.print();
    std::cout<<"============================================"<<std::endl;
}

MatSDS Twist::toMat()
{
    MatSDS mat(1,6);
    mat[0] = v[0];
    mat[1] = v[1];
    mat[2] = v[2];
    mat[3] = omg[0];
    mat[4] = omg[1];
    mat[5] = omg[2];
    return mat;
}

MatSDS Twist::toDiagMat()
{
    MatSDS mat = MatSDS::eye(6);
    mat(0,0) = v[0];
    mat(1,1) = v[1];
    mat(2,2) = v[2];
    mat(3,3) = omg[0];
    mat(4,4) = omg[1];
    mat(5,5) = omg[2];
    return mat;
}

VectorXSDS Twist::toVec()
{
    VectorXSDS vec(6);
    vec[0] = v[0];
    vec[1] = v[1];
    vec[2] = v[2];
    vec[3] = omg[0];
    vec[4] = omg[1];
    vec[5] = omg[2];
    return vec;
}

Frame Frame::SDH(double a, double alpha, double d, double theta)
{

    Frame frame;

    double ct   =   0;
    double st   =   0;
    double ca   =   0;
    double sa   =   0;

    ct = cos(theta);
    st = sin(theta);
    sa = sin(alpha);
    ca = cos(alpha);

    RotationMatDS rotMat({  ct,    -st*ca,   st*sa,
                            st,     ct*ca,  -ct*sa,
                            0,        sa,      ca  });

    TranslationDS trans(a*ct,   a*st,   d);

    frame.rotMat = rotMat;
    frame.trans  = trans;

    return frame;
}

Frame Frame::MDH(double a, double alpha, double d, double theta)
{
    Frame frame;

    double ct   =   0;
    double st   =   0;
    double ca   =   0;
    double sa   =   0;

    ct = cos(theta);
    st = sin(theta);
    sa = sin(alpha);
    ca = cos(alpha);

    RotationMatDS rotMat({   ct,       -st,     0,
                            st*ca,  ct*ca,   -sa,
                            st*sa,  ct*sa,    ca });

    TranslationDS trans(a,  -sa*d,  ca*d);

    frame.rotMat = rotMat;
    frame.trans  = trans;

    return frame;
}

Twist Frame::diff(const Frame &base2A, const Frame &base2B)
{
    TranslationDS diffP  = base2B.trans - base2A.trans;

    RotationMatDS A2B    =  base2A.rotMat.inverse()*base2B.rotMat;

    Vector3DS diffR      = base2A.rotMat*A2B.getRot();

    return Twist(diffP, diffR);
}

Twist Frame::diffRelative(const Frame &base2A, const Frame &base2B)
{
    TranslationDS diffP  = base2B.trans - base2A.trans;

    RotationMatDS A2B    =  base2A.rotMat.inverse()*base2B.rotMat;

    Vector3DS diffR      = base2A.rotMat*A2B.getRot();

    RotationMatDS aIn =  base2A.rotMat.inverse();
    return Twist(aIn*diffP, aIn*diffR);
}

}
