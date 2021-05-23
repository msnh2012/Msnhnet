
#include "Msnhnet/math/MsnhHomTransMatS.h"

namespace Msnhnet
{

void HomTransMatDS::print()
{
    std::vector<double> data = {rotMat(0,0),  rotMat(0,1), rotMat(0,2),trans[0],
                                rotMat(1,0),  rotMat(1,1), rotMat(1,2),trans[1],
                                rotMat(2,0),  rotMat(2,1), rotMat(2,2),trans[2],
                                0,0,0,1};
    MatS<4,4,double> mat(4,4,data);
    mat.print();
}

string HomTransMatDS::toString() const
{
    std::stringstream buf;

    std::vector<double> data = {rotMat(0,0),  rotMat(0,1), rotMat(0,2),trans[0],
                                rotMat(1,0),  rotMat(1,1), rotMat(1,2),trans[1],
                                rotMat(2,0),  rotMat(2,1), rotMat(2,2),trans[2],
                                0,0,0,1};
    MatS<4,4,double> mat(4,4,data);
    buf<<mat.toString();

    return buf.str();
}

string HomTransMatDS::toHtmlString() const
{
    std::stringstream buf;

    std::vector<double> data = {rotMat(0,0),  rotMat(0,1), rotMat(0,2),trans[0],
                                rotMat(1,0),  rotMat(1,1), rotMat(1,2),trans[1],
                                rotMat(2,0),  rotMat(2,1), rotMat(2,2),trans[2],
                                0,0,0,1};
    MatS<4,4,double> mat(4,4,data);
    buf<<mat.toHtmlString();
    return buf.str();
}

void HomTransMatFS::print()
{
    std::vector<float> data = {rotMat(0,0),  rotMat(0,1), rotMat(0,2),trans[0],
                               rotMat(1,0),  rotMat(1,1), rotMat(1,2),trans[1],
                               rotMat(2,0),  rotMat(2,1), rotMat(2,2),trans[2],
                               0,0,0,1};
    MatS<4,4,float> mat(4,4,data);
    mat.print();
}

string HomTransMatFS::toString() const
{
    std::stringstream buf;

    std::vector<float> data = {rotMat(0,0),  rotMat(0,1), rotMat(0,2),trans[0],
                               rotMat(1,0),  rotMat(1,1), rotMat(1,2),trans[1],
                               rotMat(2,0),  rotMat(2,1), rotMat(2,2),trans[2],
                               0,0,0,1};
    MatS<4,4,float> mat(4,4,data);
    buf<<mat.toString();

    return buf.str();
}

string HomTransMatFS::toHtmlString() const
{
    std::stringstream buf;

    std::vector<float> data = {rotMat(0,0),  rotMat(0,1), rotMat(0,2),trans[0],
                               rotMat(1,0),  rotMat(1,1), rotMat(1,2),trans[1],
                               rotMat(2,0),  rotMat(2,1), rotMat(2,2),trans[2],
                               0,0,0,1};
    MatS<4,4,float> mat(4,4,data);
    buf<<mat.toHtmlString();
    return buf.str();
}

}
