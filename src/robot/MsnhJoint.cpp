#include "Msnhnet/robot/MsnhJoint.h"

namespace Msnhnet
{

Joint::Joint(const std::string &name, const JointType &type, const double &scale, const double &offset,
             const double &inertia, const double &damping, const double &stiffness)
    :_name(name),
      _type(type),
      _scale(scale),
      _offset(offset),
      _interia(inertia),
      _damping(damping),
      _stiffness(stiffness)
{
    if(type == JOINT_ROT_XYZ || type == JOINT_TRANS_XYZ)
    {
        throw Exception(1,"Only a axis is supported",__FILE__,__LINE__,__FUNCTION__);
    }
}

Joint::Joint(const Joint::JointType &type, const double &scale, const double &offset, const double &inertia, const double &damping, const double &stiffness)
    :_name("Untitled Name"),
      _type(type),
      _scale(scale),
      _offset(offset),
      _interia(inertia),
      _damping(damping),
      _stiffness(stiffness)
{
    if(type == JOINT_ROT_XYZ || type == JOINT_TRANS_XYZ)
    {
        throw Exception(1,"Only a axis is supported",__FILE__,__LINE__,__FUNCTION__);
    }
}

const std::string &Joint::getName() const
{
    return _name;
}

const Joint::JointType &Joint::getType() const
{
    return _type;
}

const std::string Joint::getTypeName() const
{
    switch (_type)
    {
    case JOINT_FIXED:
        return "Fixed";
        break;
    case JOINT_ROT_XYZ:
        return "RotXYZ";
        break;
    case JOINT_ROT_X:
        return "RotX";
        break;
    case JOINT_ROT_Y:
        return "RotY";
        break;
    case JOINT_ROT_Z:
        return "RotZ";
        break;
    case JOINT_TRANS_XYZ:
        return "TransXYZ";
        break;
    case JOINT_TRANS_X:
        return "TransX";
        break;
    case JOINT_TRANS_Y:
        return "TransY";
        break;
    case JOINT_TRANS_Z:
        return "TransZ";
        break;
    default:
        return "Fixed";
        break;
    }
}

Vector3D Joint::getJointAxis() const
{
    switch (_type)
    {
    case JOINT_ROT_XYZ:
        return _axis;
    case JOINT_ROT_X:
        return Vector3D({1,0,0});
    case JOINT_ROT_Y:
        return Vector3D({0,1,0});
    case JOINT_ROT_Z:
        return Vector3D({0,0,1});
    case JOINT_TRANS_XYZ:
        return _axis;
    case JOINT_TRANS_X:
        return Vector3D({1,0,0});
    case JOINT_TRANS_Y:
        return Vector3D({0,1,0});
    case JOINT_TRANS_Z:
        return Vector3D({0,0,1});
    case JOINT_FIXED:
        return Vector3D({0,0,0});
    }
    return Vector3D({0,0,0});
}

const Vector3D Joint::getOrigin() const
{
    return _origin;
}

Frame Joint::getPos(const double &q) const
{
    switch (_type)
    {
    case JOINT_FIXED:
        return Frame();
        break;
    case JOINT_ROT_X:
        return Frame(Geometry::rotX(_scale*q + _offset));
        break;
    case JOINT_ROT_Y:
        return Frame(Geometry::rotY(_scale*q + _offset));
        break;
    case JOINT_ROT_Z:
        return Frame(Geometry::rotZ(_scale*q + _offset));
        break;
    case JOINT_TRANS_X:
        return Frame(Vector3D({_scale*q + _offset,0,0}));
        break;
    case JOINT_TRANS_Y:
        return Frame(Vector3D({0,_scale*q + _offset,0}));
        break;
    case JOINT_TRANS_Z:
        return Frame(Vector3D({0,0,_scale*q + _offset}));
        break;
    }

    return Frame();
}


}

