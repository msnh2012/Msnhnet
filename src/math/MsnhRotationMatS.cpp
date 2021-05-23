#include "Msnhnet/math/MsnhRotationMatS.h"

namespace Msnhnet
{

double RotationMatDS::getRotAngle(Vector3DS &axis, double eps) const
{
    double angle,x,y,z;
    double epsilon = eps;
    double epsilon2 = eps*10;

    if ((std::abs(val[1] - val[3]) < epsilon)
            && (std::abs(val[2] - val[6])< epsilon)
            && (std::abs(val[5] - val[7]) < epsilon))
    {
        if ((std::abs(val[1] + val[3]) < epsilon2)
                && (std::abs(val[2] + val[6]) < epsilon2)
                && (std::abs(val[5] + val[7]) < epsilon2)
                && (std::abs(val[0] + val[4] + val[8]-3) < epsilon2))
        {
            axis = Vector3DS(0,0,1);
            angle = 0.0;
            return angle;
        }

        angle = M_PI;
        double xx = (val[0] + 1) / 2;
        double yy = (val[4] + 1) / 2;
        double zz = (val[8] + 1) / 2;
        double xy = (val[1] + val[3]) / 4;
        double xz = (val[2] + val[6]) / 4;
        double yz = (val[5] + val[7]) / 4;

        if ((xx > yy) && (xx > zz))
        {
            x = sqrt(xx);
            y = xy/x;
            z = xz/x;
        }
        else if (yy > zz)
        {
            y = sqrt(yy);
            x = xy/y;
            z = yz/y;
        }
        else
        {
            z = sqrt(zz);
            x = xz/z;
            y = yz/z;
        }
        axis = Vector3DS(x, y, z);
        return angle;
    }

    double f = (val[0] + val[4] + val[8] - 1) / 2;
    angle = acos(std::max(-1.0, std::min(1.0, f)));

    x = (val[7] - val[5]);
    y = (val[2] - val[6]);
    z = (val[3] - val[1]);
    axis = Vector3DS(x, y, z);
    axis.normalize();
    return angle;
}

void RotationMatDS::print() const
{
    std::cout<<"{ RotMatDS: "<<std::endl;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<val[i*3 +j]<<" ";
        }
        std::cout<<std::endl;
    }

    std::cout<<";\n}"<<std::endl;
}

std::string RotationMatDS::toString() const
{

    std::stringstream buf;

    buf<<"{ RotMatDS: "<<std::endl;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<val[i*3 +j]<<" ";
        }
        buf<<std::endl;
    }

    buf<<";\n}"<<std::endl;

    return buf.str();
}

std::string RotationMatDS::toHtmlString() const
{

    std::stringstream buf;

    buf<<"{ RotMatDS: <br/>";

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<val[i*3 +j]<<" ";
        }
        buf<<std::endl;
    }

    buf<<";\n}"<<"<br/>";

    return buf.str();
}

void RotationMatFS::print()
{
    std::cout<<"{ RotMatFS: "<<std::endl;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<val[i*3 +j]<<" ";
        }
        std::cout<<std::endl;
    }

    std::cout<<";\n}"<<std::endl;
}

std::string RotationMatFS::toString() const
{

    std::stringstream buf;

    buf<<"{ RotMatFS: "<<std::endl;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<val[i*3 +j]<<" ";
        }
        buf<<std::endl;
    }

    buf<<";\n}"<<std::endl;

    return buf.str();
}

std::string RotationMatFS::toHtmlString() const
{

    std::stringstream buf;

    buf<<"{ RotMatFS: <br/>";

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<val[i*3 +j]<<" ";
        }
        buf<<std::endl;
    }

    buf<<";\n}"<<"<br/>";

    return buf.str();
}

}
