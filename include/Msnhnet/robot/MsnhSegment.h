#ifndef SEGMENT_H
#define SEGMENT_H

#include "Msnhnet/robot/MsnhFrame.h"
#include "Msnhnet/robot/MsnhJoint.h"

namespace Msnhnet
{

class MsnhNet_API Segment
{
public:
    Segment(const std::string &name, const Joint &joint=Joint(Joint::JOINT_FIXED), const Frame& endToTip=Frame());
    Segment(const Joint &joint=Joint(Joint::JOINT_FIXED), const Frame& endToTip=Frame());

    std::string getName() const;

    Joint getJoint() const;

    Frame getEndToTip() const;

    Frame getPos(const double &q) const;

private:
    std::string _name;
    Joint       _joint;
    Frame       _endToTip;
};

}
#endif 

