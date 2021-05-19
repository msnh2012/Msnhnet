#include <iostream>
#include <Msnhnet/robot/MsnhRobot.h>

using namespace Msnhnet;

int main(){
    Chain puma560;
    // a alpah d theta
    puma560.addSegments(Segment("link1",Joint(Joint::JOINT_ROT_Z),Frame::SDH(0     ,MSNH_PI_2 ,0     ,0)));
    puma560.addSegments(Segment("link2",Joint(Joint::JOINT_ROT_Z),Frame::SDH(0.4318,0         ,0     ,0)));
    puma560.addSegments(Segment("link3",Joint(Joint::JOINT_ROT_Z),Frame::SDH(0.0203,-MSNH_PI_2,0.1500,0)));
    puma560.addSegments(Segment("link4",Joint(Joint::JOINT_ROT_Z),Frame::SDH(0     ,MSNH_PI_2 ,0.4318,0)));
    puma560.addSegments(Segment("link5",Joint(Joint::JOINT_ROT_Z),Frame::SDH(0     ,-MSNH_PI_2,0     ,0)));
    puma560.addSegments(Segment("link6",Joint(Joint::JOINT_ROT_Z),Frame::SDH(0     ,0         ,0     ,0)));
	
    Frame f = ChainFK::jointToCartesian(puma560,{MSNH_PI_2,0,-MSNH_PI_2,0,0,0});
    std::cout << "Final frame: \n";
    f.print();
    std::cout << "Euler(Rad):  \n";
    Geometry::rotMat2Euler(f.getRotationMat(),ROT_ZYX).print();
    std::cout << "\nPosition:  \n";
    f.getTranslation().print();

    return 1;
}