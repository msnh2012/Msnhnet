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
    case JOINT_ROT_XYZ:
        return "RotXYZ";
    case JOINT_ROT_X:
        return "RotX";
    case JOINT_ROT_Y:
        return "RotY";
    case JOINT_ROT_Z:
        return "RotZ";
    case JOINT_TRANS_XYZ:
        return "TransXYZ";
    case JOINT_TRANS_X:
        return "TransX";
    case JOINT_TRANS_Y:
        return "TransY";
    case JOINT_TRANS_Z:
        return "TransZ";
    default:
        return "Fixed";
    }
}

Vector3DS Joint::getJointAxis() const
{
    switch (_type)
    {
    case JOINT_ROT_XYZ:
        return _axis;
    case JOINT_ROT_X:
        return Vector3DS(1,0,0);
    case JOINT_ROT_Y:
        return Vector3DS(0,1,0);
    case JOINT_ROT_Z:
        return Vector3DS(0,0,1);
    case JOINT_TRANS_XYZ:
        return _axis;
    case JOINT_TRANS_X:
        return Vector3DS(1,0,0);
    case JOINT_TRANS_Y:
        return Vector3DS(0,1,0);
    case JOINT_TRANS_Z:
        return Vector3DS(0,0,1);
    case JOINT_FIXED:
        return Vector3DS(0,0,0);
    }
    return Vector3DS(0,0,0);
}

const Vector3DS Joint::getOrigin() const
{
    return _origin;
}

Frame Joint::getPos(const double &q) const
{

    switch (_type)
    {
    case JOINT_FIXED:
        return Frame();
    case JOINT_ROT_X:
        return Frame(GeometryS::rotX(_scale*q + _offset));
    case JOINT_ROT_Y:
        return Frame(GeometryS::rotY(_scale*q + _offset));
    case JOINT_ROT_Z:
        return Frame(GeometryS::rotZ(_scale*q + _offset));
    case JOINT_TRANS_X:
        return Frame(TranslationDS(_scale*q + _offset,0,0));
    case JOINT_TRANS_Y:
        return Frame(TranslationDS(0,_scale*q + _offset,0));
    case JOINT_TRANS_Z:
        return Frame(TranslationDS(0,0,_scale*q + _offset));
    default:
        return Frame();
    }
}

Twist Joint::getTwist(const double &qdot) const
{

    switch (_type)
    {
    case JOINT_FIXED:
        return Twist();
    case JOINT_ROT_X:
        return Twist(LinearVelDS(0,0,0),AngularVelDS(_scale*qdot,0,0));
    case JOINT_ROT_Y:
        return Twist(LinearVelDS(0,0,0),AngularVelDS(0,_scale*qdot,0));
    case JOINT_ROT_Z:
        return Twist(LinearVelDS(0,0,0),AngularVelDS(0,0,_scale*qdot));
    case JOINT_TRANS_X:
        return Twist(LinearVelDS(_scale*qdot,0,0),AngularVelDS(0,0,0));
    case JOINT_TRANS_Y:
        return Twist(LinearVelDS(0,_scale*qdot,0),AngularVelDS(0,0,0));
    case JOINT_TRANS_Z:
        return Twist(LinearVelDS(0,0,_scale*qdot),AngularVelDS(0,0,0));
    default:
        return Twist();
    }
}

}

