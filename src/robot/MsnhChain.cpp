#include "Msnhnet/robot/MsnhChain.h"

namespace Msnhnet
{

Chain::Chain():
    segments(0),
    _numOfJoints(0),
    _numOfSegments(0)
{

}

Chain::Chain(const Chain &chain):
    segments(0),
    _numOfJoints(0),
    _numOfSegments(0)

{
    for(uint32_t i=0; i<chain.getNumOfSegments();i++)
    {
        addSegments(chain.getSegment(i));
    }
}

Chain &Chain::operator=(const Chain &chain)
{
    _numOfJoints    = 0;
    _numOfSegments  = 0;
    segments.clear();

    for(uint32_t i=0; i<chain.getNumOfSegments();i++)
    {
        addSegments(chain.getSegment(i));
    }
    return *this;
}

void Chain::addSegments(const Segment &segment)
{
    segments.push_back(segment);
    _numOfSegments++;
    if(segment.getJoint().getType() != Joint::JOINT_FIXED)
    {
        _numOfJoints++;
    }
}

uint32_t Chain::getNumOfJoints() const
{
    return _numOfJoints;
}

uint32_t Chain::getNumOfSegments() const
{
    return _numOfSegments;
}

const Segment &Chain::getSegment(uint32_t idx) const
{
    return segments[idx];
}

Segment &Chain::getSegment(uint32_t idx)
{
    return segments[idx];
}

}

