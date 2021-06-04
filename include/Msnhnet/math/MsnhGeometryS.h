#ifndef MSNHGEOMETRYS_H
#define MSNHGEOMETRYS_H

#include "Msnhnet/math/MsnhRotationMatS.h"
#include "Msnhnet/math/MsnhQuaternionS.h"

namespace Msnhnet
{
class MsnhNet_API GeometryS
{
public:
     static RotationMatDS euler2RotMat(const EulerDS &euler, const RotSequence& seq);
     static RotationMatFS euler2RotMat(const EulerFS &euler, const RotSequence& seq);

     static QuaternionDS  euler2Quaternion(const EulerDS& euler, const RotSequence& seq);
     static QuaternionFS  euler2Quaternion(const EulerFS& euler, const RotSequence& seq);

     static EulerDS rotMat2Euler(const RotationMatDS& rotMat, const RotSequence &seq);
     static EulerFS rotMat2Euler(const RotationMatFS& rotMat, const RotSequence &seq);

     static EulerDS quaternion2Euler(const QuaternionDS& q, const RotSequence& seq);
     static EulerFS quaternion2Euler(const QuaternionFS& q, const RotSequence& seq);

     static QuaternionDS  rotMat2Quaternion(const RotationMatDS& rotMat);
     static QuaternionFS  rotMat2Quaternion(const RotationMatFS& rotMat);

     static RotationMatDS quaternion2RotMat(const QuaternionDS& q);
     static RotationMatFS quaternion2RotMat(const QuaternionFS& q);

     static QuaternionDS  rotVec2Quaternion(const RotationVecDS& rotVec);
     static QuaternionFS  rotVec2Quaternion(const RotationVecFS& rotVec);

     static RotationVecDS quaternion2RotVec(const QuaternionDS& q);
     static RotationVecFS quaternion2RotVec(const QuaternionFS& q);

     static RotationMatDS rotVec2RotMat(const RotationVecDS& rotVec);
     static RotationMatFS rotVec2RotMat(const RotationVecFS& rotVec);

     static RotationVecDS rotMat2RotVec(const RotationMatDS& rotMat);
     static RotationVecFS rotMat2RotVec(const RotationMatFS& rotMat);

     static RotationVecDS euler2RotVec(const EulerDS& euler, const RotSequence& seq);
     static RotationVecFS euler2RotVec(const EulerFS& euler, const RotSequence& seq);

     static EulerDS rotVec2Euler(const RotationVecDS& rotVec, const RotSequence& seq);
     static EulerFS rotVec2Euler(const RotationVecFS& rotVec, const RotSequence& seq);

     static TranslationDS rotatePos(const RotationMatDS& rotMat, const TranslationDS& trans);
     static TranslationFS rotatePos(const RotationMatFS& rotMat, const TranslationFS& trans);

     inline static RotationMatDS rotZ(double angleInRad)
     {
         const double cosc  = cos(angleInRad);
         const double sinc  = sin(angleInRad);

         RotationMatDS Rz;

         Rz.val[0] = cosc;
         Rz.val[1] = -sinc;
         Rz.val[3] = sinc;
         Rz.val[4] = cosc;

         return Rz;
     }
     inline static RotationMatDS rotY(double angleInRad)
     {
         const double cosb  = cos(angleInRad);
         const double sinb  = sin(angleInRad);

         RotationMatDS Ry;

         Ry.val[0] = cosb;
         Ry.val[2] = sinb;
         Ry.val[6] = -sinb;
         Ry.val[8] = cosb;

         return Ry;
     }
     inline static RotationMatDS rotX(double angleInRad)
     {
         const double cosa  = cos(angleInRad);
         const double sina  = sin(angleInRad);
         RotationMatDS Rx;

         Rx.val[4] = cosa;
         Rx.val[5] = -sina;
         Rx.val[7] = sina;
         Rx.val[8] = cosa;

         return Rx;
     }

     inline static RotationMatFS rotZ(float angleInRad)
     {

         const float cosc  = cosf(angleInRad);
         const float sinc  = sinf(angleInRad);

         RotationMatFS Rz;

         Rz.val[0] = cosc;
         Rz.val[1] = -sinc;
         Rz.val[3] = sinc;
         Rz.val[4] = cosc;

         return Rz;
     }
     inline static RotationMatFS rotY(float angleInRad)
     {
         const float cosb  = cosf(angleInRad);
         const float sinb  = sinf(angleInRad);

         RotationMatFS Ry;
         Ry.val[0] = cosb;
         Ry.val[2] = sinb;
         Ry.val[6] = -sinb;
         Ry.val[8] = cosb;

         return Ry;
     }
     inline static RotationMatFS rotX(float angleInRad)
     {
         const float cosa  = cosf(angleInRad);
         const float sina  = sinf(angleInRad);
         RotationMatFS Rx;
         Rx.val[4] = cosa;
         Rx.val[5] = -sina;
         Rx.val[7] = sina;
         Rx.val[8] = cosa;

         return Rx;
     }

     inline static double clamp(const double &val,const double &min,const double &max)
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
     inline static float clamp(const float &val,const float &min,const float &max)
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
};

}

#endif 

