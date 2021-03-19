#include "Msnhnet/robot/MsnhChainFK.h"

namespace Msnhnet
{

Frame ChainFK::jointToCartesian(const Chain &chain, const std::vector<double> joints, int segNum)
{
    int segmentNum = 0;

    if(segNum<0)
    {
        segmentNum = chain.getNumOfSegments();
    }
    else
    {
        segmentNum = segNum;
    }

    Frame frame;

    if(joints.size()!=chain.getNumOfJoints())
    {
        throw Exception(1,"[RobotFK] input joints num != chain's joints", __FILE__, __LINE__, __FUNCTION__);
    }

    if(segmentNum > chain.getNumOfSegments())
    {
        throw Exception(1,"[RobotFK] input segments num > chain's segments", __FILE__, __LINE__, __FUNCTION__);
    }

    int jointCnt = 0;

    for (int i = 0; i < segmentNum; ++i)
    {
        if(chain.getSegment(i).getJoint().getType() != Joint::JOINT_FIXED)
        {
            frame = frame * chain.getSegment(i).getPos(joints[jointCnt]);
            jointCnt++;
        }
        else
        {
            frame = frame * chain.getSegment(i).getPos(0.0);
        }
    }

    return frame;
}

}
