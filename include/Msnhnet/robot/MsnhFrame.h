#ifndef MSNHFRAME_H
#define MSNHFRAME_H

#include "Msnhnet/cv/MsnhCVMat.h"
#include "Msnhnet/cv/MsnhCVGeometry.h"

namespace Msnhnet
{
    class Frame : public Matrix4x4D
    {
    public:
        Frame(){}
        Frame(const RotationMatD &rotMat);
        Frame(const TranslationD &trans);
        Frame(const RotationMatD &rotMat, const TranslationD &trans);

        static Frame SDH(double a,double alpha,double d,double theta);
        static Frame MDH(double a,double alpha,double d,double theta);

        QuaternionD getQuaternion() const;
    };
}

#endif // MSNHCV_H
