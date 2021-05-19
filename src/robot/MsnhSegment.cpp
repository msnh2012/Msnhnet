#include "Msnhnet/robot/MsnhSegment.h"

namespace Msnhnet
{

Segment::Segment(const std::string &name, const Joint &joint, const Frame &endToTip)
    :_name(name),
     _joint(joint),
     _endToTip(_joint.getPos(0).invert()*endToTip)
{

}

Segment::Segment(const Joint &joint, const Frame &endToTip)
    :_name("untitled segment"),
      _joint(joint),
      _endToTip(_joint.getPos(0).invert()*endToTip)
{

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

}

