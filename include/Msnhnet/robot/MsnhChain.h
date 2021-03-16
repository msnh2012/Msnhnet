#ifndef CHAIN_H
#define CHAIN_H

#include "Msnhnet/robot/MsnhSegment.h"

namespace Msnhnet
{

class Chain
{
public:
    Chain();
    Chain(const Chain& chain);
    Chain& operator= (const Chain &chain);



    std::vector<Segment> segments;
private:
    unsigned int _numOfJoints;
    unsigned int _numOfSegments;


};

}

#endif // CHAIN_H
