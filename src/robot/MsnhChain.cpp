#include "Msnhnet/robot/MsnhChain.h"

namespace Msnhnet
{

inline static double fRand(double min, double max)
{
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}

inline double minfuncSumSquared(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
  Chain *chain = (Chain *) data;

  std::vector<double> vals(x);

  double jump = 0.000001;
  double result[1];
  chain->cartSumSquaredErr(vals, result);

  if (!grad.empty())
  {
    double v1[1];
    for (unsigned int i = 0; i < x.size(); i++)
    {
      double original = vals[i];

      vals[i] = original + jump;
      chain->cartSumSquaredErr(vals, v1);

      vals[i] = original;
      grad[i] = (v1[0] - result[0]) / (2.0 * jump);
    }
  }

  return result[0];
}

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
    _minJoints.clear();
    _maxJoints.clear();
    _jointMoveTypes.clear();
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
    _minJoints.clear();
    _maxJoints.clear();
    _jointMoveTypes.clear();

    for(uint32_t i=0; i<chain.getNumOfSegments();i++)
    {
        addSegments(chain.getSegment(i));
    }
    return *this;
}

void Chain::initOpt()
{
    _opt = nlopt::opt(nlopt::LD_SLSQP, _numOfJoints);
    _opt.set_xtol_abs(0.000001);
    _opt.set_min_objective(minfuncSumSquared, this);
    _optInited = true;
}

void Chain::addSegments(const Segment &segment)
{
    segments.push_back(segment);
    _numOfSegments++;
    if(segment.getJoint().getType() != Joint::JOINT_FIXED)
    {
        _numOfJoints++;

        _minJoints.push_back(segment.getJointMin());
        _maxJoints.push_back(segment.getJointMax());
        _jointMoveTypes.push_back(segment.getMoveType());
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

MatSDS Chain::jacobi(const VectorXSDS &joints, int segNum) const
{
    int segmentNum = 0;

    if(segNum<0)
    {
        segmentNum = _numOfSegments;
    }
    else
    {
        segmentNum = segNum;
    }

    if(joints.mN!=_numOfSegments)
    {
        throw Exception(1,"[Robot Chain] input joints num != chain's joints", __FILE__, __LINE__, __FUNCTION__);
    }

    if(segmentNum > _numOfSegments)
    {
        throw Exception(1,"[Robot Chain] input segments num > chain's segments", __FILE__, __LINE__, __FUNCTION__);
    }

    Twist twist;

    Frame fk;
    Frame tTmp;

    MatSDS jac = MatSDS(6,6);

    int jointCnt = 0;

    int jacCnt   = 0;

    for (int i = 0; i < segmentNum; ++i)
    {

        if(segments[i].getJoint().getType() != Joint::JOINT_FIXED)
        {

            fk = tTmp * segments[i].getPos(joints[jointCnt]);
            twist = tTmp.rotMat * segments[i].getTwist(joints[jointCnt],1.0);
            jointCnt++;
        }
        else
        {
            fk = tTmp * segments[i].getPos(0.0);
        }
        changeRefPoint(jac, fk.trans - tTmp.trans);

        if(segments[i].getJoint().getType() != Joint::JOINT_FIXED)
        {
            jac.setCol(jacCnt++, twist.toMat());
        }

        tTmp = fk;
    }

    return jac;

}

Frame Chain::fk(const VectorXSDS &joints, int segNum)
{
    int segmentNum = 0;

    if(segNum<0)
    {
        segmentNum = _numOfSegments;
    }
    else
    {
        segmentNum = segNum;
    }

    Frame frame;

    if(joints.mN!= _numOfJoints)
    {
        throw Exception(1,"[RobotFK] input joints num != chain's joints", __FILE__, __LINE__, __FUNCTION__);
    }

    if(segmentNum > _numOfSegments)
    {
        throw Exception(1,"[RobotFK] input segments num > chain's segments", __FILE__, __LINE__, __FUNCTION__);
    }

    int jointCnt = 0;

    for (int i = 0; i < segmentNum; ++i)
    {
        if(segments[i].getJoint().getType() != Joint::JOINT_FIXED)
        {
            frame = frame * segments[i].getPos(joints[jointCnt]);
            jointCnt++;
        }
        else
        {
            frame = frame * segments[i].getPos(0.0);
        }
    }

    return frame;
}

int Chain::ikNewton(const Frame &desireFrame, VectorXSDS &outJoints, double maxIter, double eps)
{
    Frame frame;
    Twist dtTwist;

    if(outJoints.mN != _numOfJoints)
    {
        return -2;
    }

    MatSDS jac(_numOfJoints,6);

    for (int i = 0; i < maxIter; ++i)
    {

        frame   = fk(outJoints);

        dtTwist = Frame::diff(frame, desireFrame);  

        if(dtTwist.closeToEps(eps))
        {
            return i;
        }

        jac = jacobi(outJoints); 

        MatSDS dtJoints = jac.pseudoInvert()*dtTwist.toMat(); 

        for (int i = 0; i < outJoints.mN; ++i)
        {
            outJoints[i] = outJoints[i] + dtJoints[i];

            int time = (int)(outJoints[i]/MSNH_2_PI);
            outJoints[i] -= time * MSNH_2_PI;
        }

    }
    return -1;
}

int Chain::ikNewtonJL(const Frame &desireFrame, VectorXSDS &outJoints, double maxIter, double eps)
{
    Frame frame;
    Twist dtTwist;

    if(outJoints.mN != _numOfJoints)
    {
        return -2;
    }

    MatSDS jac(_numOfJoints,6);

    for (int i = 0; i < maxIter; ++i)
    {

        frame   = fk(outJoints);

        dtTwist = Frame::diff(frame, desireFrame);  

        if(dtTwist.closeToEps(eps))
        {
            return i;
        }

        jac = jacobi(outJoints); 

        auto UDVt = jac.svd();
        double sum = 0;
        VectorXSDS tmp(jac.mWidth);

        VectorXSDS dtJoints(jac.mWidth);

        for (int i = 0; i < jac.mWidth; ++i)
        {
            sum = 0;
            for (int j = 0; j < jac.mHeight; ++j)
            {
                sum +=  UDVt[0](i,j)*dtTwist[j];
            }

            if(fabs(UDVt[1][i])<eps)
            {
                tmp[i] = 0;
            }
            else
            {
                tmp[i] = sum/UDVt[1][i];
            }
        }

        for (int i = 0; i < jac.mWidth; ++i)
        {
            sum = 0;
            for (int j=0;j<jac.mWidth;j++)
            {
                sum+=UDVt[2](i,j)*tmp[j];
            }

            dtJoints[i] = sum;
        }

        for (int i = 0; i < outJoints.mN; ++i)
        {
            outJoints[i] = outJoints[i] + dtJoints[i];

            if(_jointMoveTypes[i] != Joint::TRANS_MOVE && _jointMoveTypes[i] != Joint::ROT_CONTINUOUS_MOVE)
            {
                int time = (int)(outJoints[i]/MSNH_2_PI);
                outJoints[i] -= time * MSNH_2_PI;
            }

            if(outJoints[i] < _minJoints[i])
            {
                outJoints[i] = _minJoints[i];
            }
            else if(outJoints[i] > _maxJoints[i])
            {
                outJoints[i] = _maxJoints[i];
            }
        }

    }
    return -1;
}

int Chain::ikNewtonRR(const Frame &desireFrame, VectorXSDS &outJoints, const Twist &bounds, const bool &randomStart, const bool &wrap, double maxIter, double eps, double maxTime)
{
    Frame frame;
    Twist dtTwist;

    if(outJoints.mN != _numOfJoints)
    {
        return -2;
    }

    MatSDS jac(_numOfJoints,6);

    for (int i = 0; i < maxIter; ++i)
    {

        frame   = fk(outJoints);

        dtTwist = Frame::diffRelative(desireFrame,frame);  

        if (std::abs(dtTwist.v[0]) <= std::abs(bounds.v[0]))
            dtTwist.v[0] = 0;

        if (std::abs(dtTwist.v[1]) <= std::abs(bounds.v[1]))
            dtTwist.v[1] = 0;

        if (std::abs(dtTwist.v[2]) <= std::abs(bounds.v[2]))
            dtTwist.v[2] = 0;

        if (std::abs(dtTwist.omg[0]) <= std::abs(bounds.omg[0]))
            dtTwist.omg[0] = 0;

        if (std::abs(dtTwist.omg[1]) <= std::abs(bounds.omg[1]))
            dtTwist.omg[1] = 0;

        if (std::abs(dtTwist.omg[2]) <= std::abs(bounds.omg[2]))
            dtTwist.omg[2] = 0;

        if(dtTwist.closeToEps(eps))
        {
            return i;
        }

        dtTwist = Frame::diff(frame,desireFrame);

        jac = jacobi(outJoints); 

        auto UDVt = jac.svd();
        double sum = 0;
        VectorXSDS tmp(jac.mWidth);

        VectorXSDS dtJoints(jac.mWidth);

        for (int i = 0; i < jac.mWidth; ++i)
        {
            sum = 0;
            for (int j = 0; j < jac.mHeight; ++j)
            {
                sum +=  UDVt[0](i,j)*dtTwist[j];
            }

            if(fabs(UDVt[1][i])<eps)
            {
                tmp[i] = 0;
            }
            else
            {
                tmp[i] = sum/UDVt[1][i];
            }
        }

        for (int i = 0; i < jac.mWidth; ++i)
        {
            sum = 0;
            for (int j=0;j<jac.mWidth;j++)
            {
                sum+=UDVt[2](i,j)*tmp[j];
            }

            dtJoints[i] = sum;
        }

        VectorXSDS currJoints(outJoints.mN);

        for (int i = 0; i < outJoints.mN; ++i)
        {
            currJoints[i] = outJoints[i] + dtJoints[i];

            auto mtype      = _jointMoveTypes[i];
            auto minJoint   = _minJoints[i];
            auto maxJoint   = _maxJoints[i];

            if( mtype == Joint::ROT_CONTINUOUS_MOVE)
            {
                continue;
            }

            if(currJoints[i] < minJoint)
            {
                if(!wrap ||  mtype == Joint::TRANS_MOVE)
                {
                    currJoints[i] = minJoint;
                }
                else
                {
                    double diffAngle = fmod(minJoint - currJoints[i], MSNH_2_PI);
                    double currAngle = minJoint - diffAngle + MSNH_2_PI;
                    if (currAngle > maxJoint)
                        currJoints[i] = minJoint;
                    else
                        currJoints[i] = currAngle;
                }
            }

            if(currJoints[i] > maxJoint)
            {
                if(!wrap ||  mtype == Joint::TRANS_MOVE)
                {
                    currJoints[i] = maxJoint;
                }
                else
                {
                    double diffAngle = fmod(currJoints[i] - maxJoint, MSNH_2_PI);
                    double currAngle = maxJoint + diffAngle - MSNH_2_PI;
                    if (currAngle < minJoint)
                        currJoints[i] = maxJoint;
                    else
                        currJoints[i] = currAngle;
                }
            }

            outJoints[i] = outJoints[i] - currJoints[i];
        }

        if(outJoints.isFuzzyNull(MSNH_F32_EPS))
        {
            if(randomStart)
            {
                for (unsigned int j = 0; j < currJoints.mN; j++)
                {
                    if (_jointMoveTypes[j] == Joint::ROT_CONTINUOUS_MOVE)
                        currJoints[j] = fRand(currJoints[j] - MSNH_2_PI, currJoints[j] + MSNH_2_PI);
                    else
                        currJoints[j] = fRand(_minJoints[j], _maxJoints[j]);
                }
            }
        }

        outJoints = currJoints;
    }
    return -1;
}

void Chain::cartSumSquaredErr(const std::vector<double> &x, double error[])
{
    if(_stopImmediately || _optStatus != -3)
    {
        _opt.force_stop();
        return;
    }

    Frame frame = fk(x);

    if(std::isnan(frame.trans[0]))
    {
        error[0] = std::numeric_limits<float>::max();
        _optStatus = -1;
        return;
    }

    Twist dtTwist = Frame::diffRelative(_desireFrameSQP, frame);

    if (std::abs(dtTwist.v[0]) <= std::abs(_boundsSQP.v[0]))
        dtTwist.v[0] = 0;

    if (std::abs(dtTwist.v[1]) <= std::abs(_boundsSQP.v[1]))
        dtTwist.v[1] = 0;

    if (std::abs(dtTwist.v[2]) <= std::abs(_boundsSQP.v[2]))
        dtTwist.v[2] = 0;

    if (std::abs(dtTwist.omg[0]) <= std::abs(_boundsSQP.omg[0]))
        dtTwist.omg[0] = 0;

    if (std::abs(dtTwist.omg[1]) <= std::abs(_boundsSQP.omg[1]))
        dtTwist.omg[1] = 0;

    if (std::abs(dtTwist.omg[2]) <= std::abs(_boundsSQP.omg[2]))
        dtTwist.omg[2] = 0;

    error[0] = Vector3DS::dotProduct(dtTwist.v,dtTwist.v) + Vector3DS::dotProduct(dtTwist.omg,dtTwist.omg);

    if(dtTwist.closeToEps(_epsSQP))
    {
        _optStatus = 1;
        _bestXSQP     = x;
        return;
    }
}

int Chain::ikSQPSumSqr(const Frame &desireFrame, VectorXSDS &outJoints, const Twist &bounds, double maxIter, double eps, double maxTime)
{

    if(outJoints.mN != _numOfJoints)
    {
        return -2;
    }

    _boundsSQP = bounds;

    _epsSQP    = eps;

    _opt.set_maxtime(maxTime);

    double minf; /* the minimum objective value, upon return */

    _desireFrameSQP = desireFrame;

    std::vector<double> x(_numOfJoints);

    for (unsigned int i = 0; i < x.size(); i++)
    {
      x[i] = outJoints[i];

      if (_jointMoveTypes[i] == Joint::ROT_CONTINUOUS_MOVE)
        continue;

      if (_jointMoveTypes[i] == Joint::TRANS_MOVE)
      {
        x[i] = std::min(x[i], _maxJoints[i]);
        x[i] = std::max(x[i], _minJoints[i]);
      }
      else
      {

        if (x[i] > _maxJoints[i])
        {

          double diffangle = fmod(x[i] - _maxJoints[i], MSNH_2_PI);

          x[i] = _maxJoints[i] + diffangle - MSNH_2_PI;
        }

        if (x[i] < _minJoints[i])
        {

          double diffangle = fmod(_minJoints[i] - x[i], MSNH_2_PI);

          x[i] = _minJoints[i] - diffangle + MSNH_2_PI;
        }

        if (x[i] > _maxJoints[i])
          x[i] = (_maxJoints[i] + _minJoints[i]) / 2.0;
      }
    }

    _bestXSQP   = x;
    _optStatus  = -3;

    std::vector<double> artificialLowerLimits(_minJoints.size());

    for (unsigned int i = 0; i < _minJoints.size(); i++)
    {
        if (_jointMoveTypes[i] == Joint::ROT_CONTINUOUS_MOVE)
        {
            artificialLowerLimits[i] = _bestXSQP[i] - MSNH_2_PI;
        }
        else if (_jointMoveTypes[i] == Joint::TRANS_MOVE)
        {
            artificialLowerLimits[i] = _minJoints[i];
        }
        else
        {
            artificialLowerLimits[i] = std::max(_minJoints[i], _bestXSQP[i] - MSNH_2_PI);
        }
    }

    _opt.set_lower_bounds(artificialLowerLimits);

    std::vector<double> artificialUpperLimits(_minJoints.size());

    for (unsigned int i = 0; i < _maxJoints.size(); i++)
    {
        if (_jointMoveTypes[i] == Joint::ROT_CONTINUOUS_MOVE)
        {
            artificialUpperLimits[i] = _bestXSQP[i] + MSNH_2_PI;
        }
        else if (_jointMoveTypes[i] == Joint::TRANS_MOVE)
        {
            artificialUpperLimits[i] = _maxJoints[i];
        }
        else
        {
            artificialUpperLimits[i] = std::min(_maxJoints[i], _bestXSQP[i] + MSNH_2_PI);
        }
    }

    _opt.set_upper_bounds(artificialUpperLimits);

    try
    {
      _opt.optimize(x, minf);
    }
    catch (...)
    {
    }

    if (_optStatus == -1) 

    {
        _optStatus = -3;
    }

    int q = 0;

    if (!_stopImmediately && _optStatus < 0)
    {

      for (int z = 0; z < 100; ++z)
      {

          q = z;
          if(!(!_stopImmediately && _optStatus < 0))
          {
              break;
          }

        for (unsigned int i = 0; i < x.size(); i++)
        {
            x[i] = fRand(artificialLowerLimits[i], artificialUpperLimits[i]);
        }

        try
        {
          _opt.optimize(x, minf);
        }
        catch (...) {}

        if (_optStatus == -1) 

        {
            _optStatus = -3;
        }

      }
    }

    if(q == (maxIter-1))
    {
        return -1;
    }

    for (unsigned int i = 0; i < x.size(); i++)
    {
      outJoints[i] = _bestXSQP[i];
    }

    return q;
}

void Chain::changeRefPoint(MatSDS &src, const Vector3DS &baseAB) const
{
    for (int i = 0; i < src.mHeight; ++i)
    {
        MatSDS col   = src.getCol(i);

        Twist J  = Twist(LinearVelDS(col[0], col[1], col[2]),
                AngularVelDS(col[3], col[4], col[5]));

        src.setCol(i, J.refPoint(baseAB).toMat());
    }

}

std::vector<double> Chain::getMinJoints() const
{
    return _minJoints;
}

std::vector<double> Chain::getMaxJoints() const
{
    return _maxJoints;
}

std::vector<Joint::MoveType> Chain::getJointMoveTypes() const
{
    return _jointMoveTypes;
}

}

