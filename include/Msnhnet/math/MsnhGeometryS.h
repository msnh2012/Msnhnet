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

     static RotationMatDS rotZ(double angleInRad);
     static RotationMatDS rotY(double angleInRad);
     static RotationMatDS rotX(double angleInRad);

     static RotationMatFS rotZ(float angleInRad);
     static RotationMatFS rotY(float angleInRad);
     static RotationMatFS rotX(float angleInRad);

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

