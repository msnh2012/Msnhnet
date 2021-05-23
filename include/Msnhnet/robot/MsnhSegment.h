#ifndef SEGMENT_H
#define SEGMENT_H

#include "Msnhnet/robot/MsnhFrame.h"
#include "Msnhnet/robot/MsnhJoint.h"

namespace Msnhnet
{

class MsnhNet_API Segment
{
public:
    Segment(const std::string &name, const Joint &joint=Joint(Joint::JOINT_FIXED),
            const Frame& endToTip=Frame(), const double jointMin = -DBL_MAX, const double jointMax = DBL_MAX);
    Segment(const Joint &joint=Joint(Joint::JOINT_FIXED), const Frame& endToTip=Frame(),
            const double jointMin = -DBL_MAX, const double jointMax = DBL_MAX);

    Segment(const Segment& in);
    Segment& operator=(const Segment& in);

    std::string getName() const;

    Joint getJoint() const;

    Frame getEndToTip() const;

    Frame getPos(const double &q) const;

    Twist getTwist(const double &q, const double &qdot) const;

    double getJointMin() const;

    double getJointMax() const;

    Joint::MoveType getMoveType() const;

private:
    std::string _name;
    Joint       _joint;
    Frame       _endToTip;
    double      _jointMin    =   -DBL_MAX;
    double      _jointMax    =   DBL_MAX;
    Joint::MoveType _moveType;
};

}
#endif 

