#include "Msnhnet/math/MsnhVector3S.h"

namespace Msnhnet
{

void Vector3FS::print()
{
    std::cout<<"{ Vector3FS: "<<std::endl;

    for (int i = 0; i < 3; ++i)
    {
        std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<val[i]<<" ";
    }

    std::cout<<";\n}"<<std::endl;
}

std::string Vector3FS::toString() const
{

    std::stringstream buf;

    buf<<"{ Vector3FS: "<<std::endl;
    for (int i = 0; i < 3; ++i)
    {
        buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<val[i]<<" ";
    }

    buf<<";\n}"<<std::endl;
    return buf.str();
}

std::string Vector3FS::toHtmlString() const
{

    std::stringstream buf;

    buf<<"{ Vector3FS: <br/>";

    for (int i = 0; i < 3; ++i)
    {
        buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<val[i]<<" ";
    }

    buf<<";\n}"<<"<br/>";

    return buf.str();
}

void Vector3DS::print()
{
    std::cout<<"{ Vector3DS: "<<std::endl;

    for (int i = 0; i < 3; ++i)
    {
        std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<val[i]<<" ";
    }

    std::cout<<";\n}"<<std::endl;
}

std::string Vector3DS::toString() const
{

    std::stringstream buf;

    buf<<"{ Vector3DS: "<<std::endl;
    for (int i = 0; i < 3; ++i)
    {
        buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<val[i]<<" ";
    }
    buf<<";\n}"<<std::endl;

    return buf.str();
}

std::string Vector3DS::toHtmlString() const
{

    std::stringstream buf;

    buf<<"{ Vector3DS: <br/>";

    for (int i = 0; i < 3; ++i)
    {
        buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<val[i]<<" ";
    }

    buf<<";\n}"<<"<br/>";

    return buf.str();
}

}
