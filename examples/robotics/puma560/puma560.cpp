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

    VectorXSDS q({ MSNH_PI_2,MSNH_PI_3,MSNH_PI_4,MSNH_PI_6,MSNH_PI,MSNH_PI_2 });


    // ================ fk ===============
    Frame frame = puma560.fk(q);
    std::cout << "Final frame: \n";
    frame.print();
    std::cout << "Euler(Deg):  \n";
    (GeometryS::rotMat2Euler(frame.rotMat, ROT_ZYX)*180/MSNH_PI).print();


    // ================ ik ===============
    VectorXSDS outq(6);
    int res = puma560.ikNewton(frame, outq);

    std::cout << std::endl;
    std::cout << "==============================================================" << std::endl;
    std::cout << "Newton Raphson: " << (res > 0 ? "succeed" : "failed") << "  iters: " << res << std::endl;
    outq.print();
    std::cout << "Newton Raphson: fk check:" << std::endl;
    puma560.fk(outq).print();

    outq.fill(0);
    res = puma560.ikNewtonJL(frame, outq);

    std::cout << std::endl;
    std::cout << "==============================================================" << std::endl;
    std::cout << "Newton Raphson joint limits: " << (res > 0 ? "succeed" : "failed") << "  iters: " << res << std::endl;
    outq.print();
    std::cout << "Newton Raphson joint limits: fk check:" << std::endl;
    puma560.fk(outq).print();

    outq.fill(0);
    res = puma560.ikNewtonRR(frame, outq);
    std::cout << std::endl;
    std::cout << "==============================================================" << std::endl;
    std::cout << "Newton Raphson random start: " << (res > 0 ? "succeed" : "failed") << "  iters: " << res << std::endl;
    outq.print();
    std::cout << "Newton Raphson random start: fk check:" << std::endl;
    puma560.fk(outq).print();

    outq.fill(0);
    puma560.initOpt();
    res = puma560.ikSQPSumSqr(frame, outq);
    std::cout << std::endl;
    std::cout << "==============================================================" << std::endl;
    std::cout << "SQP sum squared: " << (res >= 0 ? "succeed" : "failed") << "  iters: " << res << std::endl;
    outq.print();
    std::cout << "SQP sum squared: fk check:" << std::endl;
    puma560.fk(outq).print();

    getchar();

    return 1;
}
