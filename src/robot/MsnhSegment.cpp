#include "Msnhnet/robot/MsnhSegment.h"

namespace Msnhnet
{

Segment::Segment(const std::string &name, const Joint &joint, const Frame &endToTip, const double jointMin , const double jointMax)
    :_name(name),
     _joint(joint),
     _jointMin(jointMin),
     _jointMax(jointMax),
     _endToTip(_joint.getPos(0).invert()*endToTip)
{
    if(joint.getType() == Joint::JOINT_FIXED)
    {
        _moveType = Joint::CAN_NOT_MOVE;
    }
    else if( joint.getType()>=Joint::JOINT_ROT_XYZ && joint.getType()<Joint::JOINT_TRANS_XYZ)
    {

        if(abs(_jointMin+DBL_MAX) < MSNH_F64_EPS && abs(_jointMin-DBL_MAX) < MSNH_F64_EPS)
        {
            _moveType = Joint::ROT_CONTINUOUS_MOVE;
        }
        else
        {
            _moveType = Joint::ROT_LIMIT_MOVE;
        }
    }
    else
    {
        _moveType = Joint::TRANS_MOVE;
    }
}

Segment::Segment(const Joint &joint, const Frame &endToTip,  const double jointMin , const double jointMax)
    :_name("untitled segment"),
      _joint(joint),
      _jointMin(jointMin),
      _jointMax(jointMax),
      _endToTip(_joint.getPos(0).invert()*endToTip)
{
    if(joint.getType() == Joint::JOINT_FIXED)
    {
        _moveType = Joint::CAN_NOT_MOVE;
    }
    else if( joint.getType()>=Joint::JOINT_ROT_XYZ && joint.getType()<Joint::JOINT_TRANS_XYZ)
    {

        if(abs(_jointMin+DBL_MAX) < MSNH_F64_EPS && abs(_jointMin-DBL_MAX) < MSNH_F64_EPS)
        {
            _moveType = Joint::ROT_CONTINUOUS_MOVE;
        }
        else
        {
            _moveType = Joint::ROT_LIMIT_MOVE;
        }
    }
    else
    {
        _moveType = Joint::TRANS_MOVE;
    }
}

Segment::Segment(const Segment &in):
    _name(in._name),
    _joint(in._joint),
    _endToTip(in._endToTip),
    _jointMin(in._jointMin),
    _jointMax(in._jointMax),
    _moveType(in._moveType)
{

}

Segment &Segment::operator=(const Segment &in)
{
    _name       = in._name;
    _joint      = in._joint;
    _endToTip   = in._endToTip;
    _jointMin   = in._jointMin;
    _jointMax   = in._jointMax;
    _moveType   = in._moveType;
    return *this;
}

std::string Segment::getName() const
{
    return _name;
}

Joint Segment::getJoint() const
{
    return _joint;
}

Frame Segment::getEndToTip() const
{
    return Frame(_joint.getPos(0)*_endToTip);
}

Frame Segment::getPos(const double &q) const
{
    return Frame(_joint.getPos(q)*_endToTip);
}

Twist Segment::getTwist(const double &q, const double &qdot) const
{
    return _joint.getTwist(qdot).refPoint(_joint.getPos(q).rotMat*_endToTip.trans);
}

double Segment::getJointMin() const
{
    return _jointMin;
}

double Segment::getJointMax() const
{
    return _jointMax;
}

Joint::MoveType Segment::getMoveType() const
{
    return _moveType;
}

}

