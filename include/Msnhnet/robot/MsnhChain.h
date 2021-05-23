﻿#ifndef CHAIN_H
#define CHAIN_H

#include "Msnhnet/robot/MsnhSegment.h"
#include "Msnhnet/3rdparty/nlopt/nlopt.hpp"

namespace Msnhnet
{

class MsnhNet_API Chain
{
public:
    std::vector<Segment> segments;
    Chain();
    Chain(const Chain& chain);
    Chain& operator= (const Chain &chain);

    void initOpt();

    void addSegments(const Segment &segment);

    uint32_t getNumOfJoints() const;

    uint32_t getNumOfSegments() const;

    const Segment& getSegment(uint32_t idx) const;

    void changeRefPoint(MatSDS &src, const Vector3DS& baseAB) const;

    std::vector<double> getMinJoints() const;

    std::vector<double> getMaxJoints() const;

    std::vector<Joint::MoveType> getJointMoveTypes() const;

    MatSDS jacobi(const VectorXSDS &joints, int segNum = -1) const;

    Frame fk(const VectorXSDS &joints, int segNum = -1);

    int ikNewton(const Frame &desireFrame, VectorXSDS &outJoints, double maxIter=100, double eps = 0.0001);

    int ikNewtonJL(const Frame &desireFrame, VectorXSDS &outJoints, double maxIter=100, double eps = 0.0001);

    int ikNewtonRR(const Frame &desireFrame, VectorXSDS &outJoints, const Twist& bounds = Twist(),
                          const bool &randomStart = true, const bool &wrap = true, double maxIter=100, double eps = 0.0001, double maxTime = 0.005);

    void cartSumSquaredErr(const std::vector<double>& x, double error[]);
    int ikSQPSumSqr(const Frame &desireFrame, VectorXSDS &outJoints, const Twist& bounds = Twist(),
                     double maxIter=100, double eps = 0.0001, double maxTime = 0.001);

private:
    nlopt::opt _opt;
    bool _optInited       = false;
    int  _optStatus       = -1;
    bool _stopImmediately = false;
    int _numOfJoints;
    int _numOfSegments;
    std::vector<double> _minJoints;
    std::vector<double> _maxJoints;
    std::vector<Joint::MoveType> _jointMoveTypes;
    double _epsSQP = 0.0001;
    Frame _desireFrameSQP;
    std::vector<double> _bestXSQP;
    Twist _boundsSQP;
};

}

#endif 

