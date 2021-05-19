#include <iostream>
#include <Msnhnet/robot/MsnhRobot.h>

using namespace Msnhnet;

void space()
{
    SE3D M;

    M.setVal({1,0,0,-817.25,
              0,0,-1,-191.45,
              0,1,0,-5.491,
              0,0,0,1
             });

    std::vector<ScrewD> screwList;
    screwList.push_back(ScrewD(Vector3D({0, 0,    0}),Vector3D({0, 0,  1})));
    screwList.push_back(ScrewD(Vector3D({89.159, 0,    0}),Vector3D({0, -1,  0})));
    screwList.push_back(ScrewD(Vector3D({89.159, 0, 425}),Vector3D({0, -1, 0})));
    screwList.push_back(ScrewD(Vector3D({89.159, 0, 817.25}),Vector3D({0, -1, 0})));
    screwList.push_back(ScrewD(Vector3D({109.15, -817.25, 0}),Vector3D({0, 0, -1})));
    screwList.push_back(ScrewD(Vector3D({-5.491, 0, 817.25}),Vector3D({0, -1, 0})));

    Vector6D thetalist0 = Vector6D({MSNH_PI_2, MSNH_PI_3,MSNH_PI_3,MSNH_PI_6,MSNH_PI_6,MSNH_PI_3});

    std::cout<<"Input joint angle: "<<std::endl;
    (180/MSNH_PI*thetalist0).print();

    ModernRobot<6> robot(screwList,M,Msnhnet::ROBOT_SPACE);

    std::cout<<"FK euler: "<<std::endl;
    (180/MSNH_PI*Msnhnet::Geometry::rotMat2Euler(robot.fk(thetalist0).getRotationMat(), ROT_ZYX)).print();
    std::cout<<"FK HomTransMatrix: "<<std::endl;
    robot.fk(thetalist0).print();

    auto T = robot.fk(thetalist0);

    Vector6D thetalist1 = Vector6D({0,0,0,0,0,0});
    bool ik = robot.ik(T,thetalist1,0.01,0.001);

    std::cout<<"IK joint angle: "<<std::endl;
    (180/MSNH_PI*thetalist1).print();
    std::cout<<"FK use IK joint angle: "<<std::endl;
    robot.fk(thetalist1).print();
}

void body()
{
    SE3D M;

    M.setVal({1,0,0,-817.25,
              0,0,-1,-191.45,
              0,1,0,-5.491,
              0,0,0,1
             });

    std::vector<ScrewD> screwList;
    screwList.push_back(ScrewD(Vector3D({191.45, 0,    817.25}),Vector3D({0, 1,  0})));
    screwList.push_back(ScrewD(Vector3D({94.65,-817.25,    0}),Vector3D({0, 0,  1})));
    screwList.push_back(ScrewD(Vector3D({94.65,-392.25,    0}),Vector3D({0, 0, 1})));
    screwList.push_back(ScrewD(Vector3D({94.65, 0, 0}),Vector3D({0,0,1})));
    screwList.push_back(ScrewD(Vector3D({-82.3, 0, 0}),Vector3D({0, -1, 0})));
    screwList.push_back(ScrewD(Vector3D({0, 0, 0}),Vector3D({0, 0, 1})));

    Vector6D thetalist0 = Vector6D({MSNH_PI_2, MSNH_PI_3,MSNH_PI_3,MSNH_PI_6,MSNH_PI_6,MSNH_PI_3});

    std::cout<<"Input joint angle: "<<std::endl;
    (180/MSNH_PI*thetalist0).print();

    ModernRobot<6> robot(screwList,M,Msnhnet::ROBOT_BODY);

    std::cout<<"FK euler: "<<std::endl;
    (180/MSNH_PI*Msnhnet::Geometry::rotMat2Euler(robot.fk(thetalist0).getRotationMat(), ROT_ZYX)).print();
    std::cout<<"FK HomTransMatrix: "<<std::endl;
    robot.fk(thetalist0).print();

    auto T = robot.fk(thetalist0);

    Vector6D thetalist1 = Vector6D({0,0,0,0,0,0});
    bool ik = robot.ik(T,thetalist1,0.01,0.001);

    std::cout<<"IK joint angle: "<<std::endl;
    (180/MSNH_PI*thetalist1).print();
    std::cout<<"FK use IK joint angle: "<<std::endl;
    robot.fk(thetalist1).print();
}

int main()
{

    std::cout << "============================== space  ==============================\n";
    space();

    std::cout << "============================== body  ==============================\n";
    body();

    return 1;
}