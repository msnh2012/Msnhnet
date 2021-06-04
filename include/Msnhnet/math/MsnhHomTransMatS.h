#ifndef MSNHHOMTRANSMATS_H
#define MSNHHOMTRANSMATS_H

#include "Msnhnet/math/MsnhGeometryS.h"
#include "Msnhnet/math/MsnhMatrixS.h"
namespace  Msnhnet
{
class MsnhNet_API HomTransMatDS
{
public:
    RotationMatDS rotMat;
    Vector3DS trans;

    HomTransMatDS(){}

    inline HomTransMatDS(const RotationMatDS& rotMat, const Vector3DS& trans)
    {
        this->rotMat = rotMat;
        this->trans  = trans;
    }

    inline HomTransMatDS(const RotationMatDS& rotMat)
    {
        this->rotMat = rotMat;
    }

    inline HomTransMatDS(const Vector3DS& trans)
    {
        this->trans  = trans;
    }

    inline HomTransMatDS& operator =(const HomTransMatDS& mat)
    {
        if(this!=&mat)
        {
            rotMat = mat.rotMat;
            trans  = mat.trans;
        }
        return *this;
    }

    inline void translate(const TranslationDS& vector)
    {
        trans[0] += vector[0];
        trans[1] += vector[1];
        trans[2] += vector[2];
    }

    inline void translate(const double &x, const double &y, const double &z)
    {
        trans[0] += x;
        trans[1] += y;
        trans[2] += z;
    }

    inline void rotate(const double &angleInRad, const double &x, const double &y, const double &z)
    {
        rotate(angleInRad,Vector3DS(x,y,z));
    }

    inline void rotate(const double &angleInRad, const Vector3DS& vector)
    {
        Vector3DS vec = vector;
        vec.normalize();
        const double x = vec[0];
        const double y = vec[1];
        const double z = vec[2];

        rotMat = GeometryS::euler2RotMat(EulerDS(x*angleInRad,y*angleInRad,z*angleInRad),RotSequence::ROT_ZYX);
    }

    inline void rotate(const EulerDS &euler)
    {
        rotMat = GeometryS::euler2RotMat(euler,RotSequence::ROT_ZYX);
    }

    inline void rotate(const QuaternionDS &quat)
    {
        rotMat = GeometryS::quaternion2RotMat(quat);
    }

    inline HomTransMatDS invert() const
    {
        HomTransMatDS tmp;

        tmp.rotMat = rotMat.inverse();
        tmp.trans  = rotMat.invMul(trans*-1);

        return tmp;
    }

     inline friend HomTransMatDS operator *(const HomTransMatDS& A, const HomTransMatDS& B)
    {
        return HomTransMatDS(A.rotMat*B.rotMat, A.rotMat*B.trans+A.trans);
    }

    void print();

    std::string toString() const;

    std::string toHtmlString() const;

    inline bool operator == (const HomTransMatDS& A)
    {
        return (rotMat == A.rotMat) && (trans == A.trans);
    }

    inline bool operator != (const HomTransMatDS& A)
    {
        return (rotMat != A.rotMat) || (trans != A.trans);
    }
};

class MsnhNet_API HomTransMatFS
{
public:
    RotationMatFS rotMat;
    Vector3FS trans;

    HomTransMatFS(){}

    inline HomTransMatFS(const RotationMatFS& rotMat, const Vector3FS& trans)
    {
        this->rotMat = rotMat;
        this->trans  = trans;
    }

    inline HomTransMatFS(const RotationMatFS& rotMat)
    {
        this->rotMat = rotMat;
    }

    inline HomTransMatFS(const Vector3FS& trans)
    {
        this->trans  = trans;
    }

    inline HomTransMatFS& operator =(const HomTransMatFS& mat)
    {
        if(this!=&mat)
        {
            rotMat = mat.rotMat;
            trans  = mat.trans;
        }
        return *this;
    }

    inline void translate(const TranslationFS& vector)
    {
        trans[0] += vector[0];
        trans[1] += vector[1];
        trans[2] += vector[2];
    }

    inline void translate(const float &x, const float &y, const float &z)
    {
        trans[0] += x;
        trans[1] += y;
        trans[2] += z;
    }

    inline void rotate(const float &angleInRad, const float &x, const float &y, const float &z)
    {
        rotate(angleInRad,Vector3FS(x,y,z));
    }

    inline void rotate(const float &angleInRad, const Vector3FS& vector)
    {
        Vector3FS vec = vector;
        vec.normalize();
        const float x = vec[0];
        const float y = vec[1];
        const float z = vec[2];

        rotMat = GeometryS::euler2RotMat(EulerFS(x*angleInRad,y*angleInRad,z*angleInRad),RotSequence::ROT_ZYX);
    }

    inline void rotate(const EulerFS &euler)
    {
        rotMat = GeometryS::euler2RotMat(euler,RotSequence::ROT_ZYX);
    }

    inline void rotate(const QuaternionFS &quat)
    {
        rotMat = GeometryS::quaternion2RotMat(quat);
    }

    inline HomTransMatFS invert() const
    {
        HomTransMatFS tmp;

        tmp.rotMat = rotMat.inverse();
        tmp.trans  = rotMat.invMul(trans*-1);
        return tmp;
    }

    inline friend HomTransMatFS operator *(const HomTransMatFS& A, const HomTransMatFS& B)
    {
        return HomTransMatFS(A.rotMat*B.rotMat, A.rotMat*B.trans+A.trans);
    }

    void print();

    std::string toString() const;

    std::string toHtmlString() const;

    inline bool operator == (const HomTransMatFS& A)
    {
        return (rotMat == A.rotMat) && (trans == A.trans);
    }

    inline bool operator != (const HomTransMatFS& A)
    {
        return (rotMat != A.rotMat) || (trans != A.trans);
    }
};

}

#endif 

