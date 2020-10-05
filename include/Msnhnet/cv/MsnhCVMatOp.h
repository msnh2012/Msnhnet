#ifndef MSNHCVOP_H
#define MSNHCVOP_H

#include <Msnhnet/cv/MsnhCVMat.h>

namespace Msnhnet
{

class MatOp
{
public:
    static void roi(const Mat &src, Mat &dst, const Vec2I32 &p1, const Vec2I32 &p2);
    static void cvtColor(const Mat &src, Mat &dst, const Vec2I32 &p1, const Vec2I32 &p2);
};

}

#endif 

