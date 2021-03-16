#ifndef JOINT_H
#define JOINT_H

#include <string>
#include "Msnhnet/robot/MsnhFrame.h"
namespace Msnhnet
{
class Joint
{
public:

    enum JointType
    {
        JOINT_FIXED,
        JOINT_ROT_XYZ,
        JOINT_ROT_X,
        JOINT_ROT_Y,
        JOINT_ROT_Z,
        JOINT_TRANS_XYZ,
        JOINT_TRANS_X,
        JOINT_TRANS_Y,
        JOINT_TRANS_Z
    };

    Joint(const std::string &name, const JointType &type, const double &scale=1, const double &offset=0,
          const double& inertia=0, const double& damping=0, const double& stiffness=0);

    Joint(const JointType &type, const double &scale=0, const double &offset=0,
          const double &inertia=0, const double &damping=0, const double &stiffness=0);

    ~Joint(){}

    const std::string  &getName() const;

    const JointType &getType() const;

    const std::string getTypeName() const;

    Vector3D getJointAxis() const;

    const Vector3D getOrigin() const;

    Frame getPos(const double &q) const;

private:
    std::string _name;
    JointType   _type;

    double      _scale      = 0;//(默认：1)输入参数和实际几何之间的缩放关系
    double      _offset     = 0;//(默认：0)输入参数和实际几何之间灯偏移
    double      _interia    = 0;//(默认：0)关节惯性
    double      _damping    = 0;//(默认：0)关节阻尼
    double      _stiffness  = 0;//(默认：0)关节刚度


    Vector3D    _axis;
    Vector3D    _origin;
    Frame       _jointPos;
};

}

#endif // JOINT_H
