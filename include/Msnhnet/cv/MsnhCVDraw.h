#ifndef MSNHCVDRAW_H
#define MSNHCVDRAW_H

#include <Msnhnet/cv/MsnhCVMat.h>

namespace Msnhnet
{
class Draw
{
public:
    static void drawLine(Mat &mat, Vec2I32 p1, Vec2I32 p2, const Vec3U8 &color);
    static void drawRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec3U8 &color);
    static void fillRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec3U8 &color);
    static void drawRect(Mat &mat, const Vec2I32 &pos, const uint32_t &width, const uint32_t& height, const Vec3U8 &color);
    static void drawRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec2I32 &p3, const Vec2I32 &p4, const Vec3U8 &color);
};
}

#endif 

