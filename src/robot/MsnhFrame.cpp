#include "Msnhnet/robot/MsnhFrame.h"

namespace Msnhnet
{

Frame::Frame(const RotationMatD &rotMat)
{
    setRotationMat(rotMat);
}

Frame::Frame(const TranslationD &trans)
{
    setTranslation(trans);
}

Frame::Frame(const RotationMatD &rotMat, const TranslationD &trans)
{
    setRotationMat(rotMat);
    setTranslation(trans);
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

    RotationMatD rotMat({   ct,    -st*ca,   st*sa,
                            st,     ct*ca,  -ct*sa,
                            0,        sa,      ca  });

    TranslationD trans({a*ct,   a*st,   d});

    frame.setRotationMat(rotMat);
    frame.setTranslation(trans);

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

    RotationMatD rotMat({   ct,       -st,     0,
                            st*ca,  ct*ca,   -sa,
                            st*sa,  ct*sa,    ca });

    TranslationD trans({a,  -sa*d,  ca*d});

    frame.setRotationMat(rotMat);
    frame.setTranslation(trans);

    return frame;
}

QuaternionD Frame::getQuaternion() const
{
    return Geometry::rotMat2Quaternion(getRotationMat());
}

}

