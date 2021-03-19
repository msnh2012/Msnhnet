#ifndef CHAIN_H
#define CHAIN_H

#include "Msnhnet/robot/MsnhSegment.h"

namespace Msnhnet
{

class MsnhNet_API Chain
{
public:
    Chain();
    Chain(const Chain& chain);
    Chain& operator= (const Chain &chain);

    void addSegments(const Segment &segment);

    std::vector<Segment> segments;

    uint32_t getNumOfJoints() const;

    uint32_t getNumOfSegments() const;

    const Segment& getSegment(uint32_t idx) const;

    Segment& getSegment(uint32_t idx);

private:
    uint32_t _numOfJoints;
    uint32_t _numOfSegments;

};

}

#endif 

