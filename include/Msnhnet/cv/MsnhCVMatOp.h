#ifndef MSNHCVOP_H
#define MSNHCVOP_H

#include <Msnhnet/cv/MsnhCVMat.h>

namespace Msnhnet
{

class MatOp
{
public:
    static void roi(Mat &src, Mat &dst, const Vec2I32 &p1, const Vec2I32 &p2);
    static void cvtColor(Mat &src, Mat &dst, const CvtColorType& cvtType);

private:
    static void RGB2BGR(const Mat &src, Mat &dst);
    static void RGB2GRAY(Mat &src, Mat &dst);
    static void RGBA2GRAY(Mat &src, Mat &dst);
};

}

#endif 

