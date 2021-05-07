#ifndef MODERNROBOT_H
#define MODERNROBOT_H

#include <Msnhnet/robot/MsnhSpatialMath.h>

namespace Msnhnet
{

enum RobotType
{
    ROBOT_SPACE,
    ROBOT_BODY
};

template<int jointNum>
class ModernRobot
{
public:
    ModernRobot(const std::vector<ScrewD> &screwList, const SE3D &M, RobotType robType=ROBOT_SPACE)
    {
        if(screwList.size()!=jointNum)
        {
            throw Exception(1, "[ModernRobot] screw list != robot joint num", __FILE__, __LINE__,__FUNCTION__);
        }

        _screwList  = screwList;
        _M          = M;
        _robType    = robType;
    }
    ~ModernRobot(){}

    SE3D fk(const Vector<jointNum,double> &joints)
    {
        SE3D T = _M;

        if(_robType==ROBOT_SPACE)
        {

            for (int i = jointNum-1; i >= 0; i--)
            {
                if(abs(joints[i])<MSNH_F64_EPS)
                {
                    continue;
                }
                T = SE3D::exp(_screwList[i]*joints[i])*T;
            }
        }
        else
        {

            for (int i = 0; i < jointNum; i++)
            {
                if(abs(joints[i])<MSNH_F64_EPS)
                {
                    continue;
                }
                T = T*SE3D::exp(_screwList[i]*joints[i]);
            }
        }
        return T;
    }

    Mat_<jointNum,6,double> jacobi(const Vector<jointNum,double> &joints)
    {
        Mat_<jointNum,6,double> J;

        if(_robType==ROBOT_SPACE)
        {
            J.setCol(0,_screwList[0].toMat());

            SE3D T = Mat::eye(4,MAT_GRAY_F64);

            for (int i = 1; i < jointNum; i++)
            {
                T = T*SE3D::exp(_screwList[i-1]*joints[i-1]);
                Mat JTmp = T.adjoint()*_screwList[i].toMat();
                J.setCol(i,JTmp);
            }
        }
        else
        {
            J.setCol(jointNum-1,_screwList[jointNum-1].toMat());

            SE3D T = Mat::eye(4,MAT_GRAY_F64);
            for (int i = jointNum-2; i >= 0; i--)
            {
                T = T*SE3D::exp(_screwList[i+1]*joints[i+1]*-1);
                Mat JTmp = T.adjoint()*_screwList[i].toMat();
                J.setCol(i,JTmp);
            }
        }
        return J;
    }

    bool ik(const SE3D& Td, Vector<jointNum,double> &joint0, double eOmg, double eV)
    {
        bool ok = false;

        if(_robType==ROBOT_SPACE)
        {
            SE3D T0 = fk(joint0);
            ScrewD Vb = ((SE3D)(T0.fastInvert()*Td)).log();
            Vector<6,double> Vs1 = ((Mat_<6,6,double>)(T0.adjoint())).mulVec(Vb.toVector());
            ScrewD Vs(Vs1);

            ok =!(Vs.v.length() > eV || Vs.w.length() > eOmg);

            int i = 0;

            while (!ok && i < _ikMaxIter)
            {
               i++;
               Mat_<6,jointNum,double> jacPinv = jacobi(joint0).pseudoInvert();
               joint0 = joint0 + jacPinv.mulVec(Vs.toVector());
               for (int i = 0; i < jointNum; ++i)
               {
                    int time = (int)(joint0[i]/MSNH_2_PI);
                    joint0[i] -= time * MSNH_2_PI;
               }
               T0 = fk(joint0);
               Vb = ((SE3D)(T0.fastInvert()*Td)).log();
               Vs1 = ((Mat_<6,6,double>)(T0.adjoint())).mulVec(Vb.toVector());
               Vs.fromVector(Vs1);
               ok =!(Vs.v.length() > eV || Vs.w.length() > eOmg);
            }
            std::cout<<"iter: "<<i<<std::endl;
            return ok;
        }
        else
        {
            SE3D T0 = fk(joint0);
            ScrewD Vb = ((SE3D)(T0.fastInvert()*Td)).log();

            ok =!(Vb.v.length() > eV || Vb.w.length() > eOmg);

            int i = 0;

            while (!ok && i < _ikMaxIter)
            {
               i++;
               Mat_<6,jointNum,double> jacPinv = jacobi(joint0).pseudoInvert();
               joint0 = joint0 + jacPinv.mulVec(Vb.toVector());
               for (int i = 0; i < jointNum; ++i)
               {
                    int time = (int)(joint0[i]/MSNH_2_PI);
                    joint0[i] -= time * MSNH_2_PI;
               }
               T0 = fk(joint0);
               Vb = ((SE3D)(T0.fastInvert()*Td)).log();
               ok =!(Vb.v.length() > eV || Vb.w.length() > eOmg);
            }
            return ok;
        }
    }

    int setIkMaxIter() const
    {
        return _ikMaxIter;
    }

    void setIkMaxIter(int ikMaxIter)
    {
        _ikMaxIter = ikMaxIter;
    }

private:
    std::vector<ScrewD> _screwList;
    SE3D _M;
    RobotType _robType;
    int _ikMaxIter    =   10000;
};
}

#endif 

