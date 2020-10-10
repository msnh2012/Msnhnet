#ifndef MSNHCVDRAW_H
#define MSNHCVDRAW_H

#include <Msnhnet/cv/MsnhCVMat.h>
#include <vector>
#include <math.h>
#include <Msnhnet/cv/MsnhCVFont.h>

namespace Msnhnet
{
class Draw
{
public:
    static void drawLine(Mat &mat, Vec2I32 p1, Vec2I32 p2, const Vec3U8 &color);
    static void drawLine(Mat &mat, Vec2I32 p1, Vec2I32 p2, const Vec3U8 &color, const int &width);
    static void drawRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec3U8 &color, const int &width);
    static void drawRect(Mat &mat, const Vec2I32 &pos, const int32_t &width, const int32_t &height, const Vec3U8 &color, const int &lineWidth);
    static void drawRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec2I32 &p3, const Vec2I32 &p4, const Vec3U8 &color, const int &width);
    static void drawPoly(Mat &mat, std::vector<Vec2I32> &points, const Vec3U8 &color, const int &width);
    static void drawEllipse(Mat &mat,const Vec2I32 &pos, const int32_t &width, const int32_t& height, const Vec3U8 &color);
    static void drawEllipse(Mat &mat,const Vec2I32 &pos, const int32_t &width, const int32_t& height, const Vec3U8 &color, const int& lineWidth);

    static void fillRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec3U8 &color, const bool &addMode=false);
    static void fillEllipse(Mat &mat,const Vec2I32 &pos, const int32_t &width, const int32_t& height, const Vec3U8 &color);

    static void drawFont(Mat &mat, const std::string& content, const Vec2I32 &pos, const Vec3U8 &color);

    static void checkMat(Mat &mat);
};
}

#endif 

