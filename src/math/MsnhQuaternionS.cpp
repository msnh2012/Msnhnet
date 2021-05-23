#include "Msnhnet/math/MsnhQuaternionS.h"

namespace Msnhnet
{

void QuaternionDS::print()
{
    std::cout<<"{ QuaternionDS: \n"<<std::endl;
    std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<q0<<" "<<q1<<" "<<q2<<" "<<q3<<" \n";
    std::cout<<"}\n"<<std::endl;
}

std::string QuaternionDS::toString()
{
    std::stringstream buf;
    buf<<"{ QuaternionDS: \n"<<std::endl;
    buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<q0<<" "<<q1<<" "<<q2<<" "<<q3<<" \n";
    buf<<"}\n"<<std::endl;
    return buf.str();
}

std::string QuaternionDS::toHtmlString()
{
    std::stringstream buf;
    buf<<"{ QuaternionDS: \n"<<"<br/>";
    buf<<std::setiosflags(std::ios::left)<<std::setprecision(12)<<std::setw(12)<<q0<<" "<<q1<<" "<<q2<<" "<<q3<<" \n";
    buf<<"}\n"<<"<br/>";
    return buf.str();
}

void QuaternionFS::print()
{
    std::cout<<"{ QuaternionFS: \n"<<std::endl;
    std::cout<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<q0<<" "<<q1<<" "<<q2<<" "<<q3<<" \n";
    std::cout<<"}\n"<<std::endl;
}

std::string QuaternionFS::toString()
{
    std::stringstream buf;
    buf<<"{ QuaternionFS: \n"<<std::endl;
    buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<q0<<" "<<q1<<" "<<q2<<" "<<q3<<" \n";
    buf<<"}\n"<<std::endl;
    return buf.str();
}

std::string QuaternionFS::toHtmlString()
{
    std::stringstream buf;
    buf<<"{ QuaternionFS: \n"<<"<br/>";
    buf<<std::setiosflags(std::ios::left)<<std::setprecision(6)<<std::setw(6)<<q0<<" "<<q1<<" "<<q2<<" "<<q3<<" \n";
    buf<<"}\n"<<"<br/>";
    return buf.str();
}

}
