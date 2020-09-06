#ifndef MSNHTYPES_H
#define MSNHTYPES_H
#include <stdint.h>
#include <algorithm>
namespace Msnhnet
{
union UInt16
{
    uint8_t  bytes[2];
    uint16_t val;
};

union Int16
{
    uint8_t  bytes[2];
    int16_t  val;
};

union UInt32
{
    uint8_t  bytes[4];
    uint32_t val;
};

union Int32
{
    uint8_t  bytes[4];
    int32_t  val;
};

union UInt64
{
    uint8_t  bytes[8];
    uint64_t val;
};

union Int64
{
    uint8_t  bytes[8];
    int64_t  val;
};

union Float32
{
    uint8_t  bytes[4];
    float    val;
};

union Float64
{
    uint8_t  bytes[8];
    double   val;
};

struct Point2I
{
    Point2I(int x, int y):x(x),y(y){}
    int x = 0;
    int y = 0;
};

class Box
{
public:
    struct XYWHBox
    {
        XYWHBox(){}
        XYWHBox(const float& x, const float& y, const float& w, const float& h)
            :x(x),y(y),w(w),h(h){}

        float    x      =   0;
        float    y      =   0;
        float    w      =   0;
        float    h      =   0;
    };

    struct X1Y1X2Y2Box
    {
        X1Y1X2Y2Box(){}
        X1Y1X2Y2Box(const float& x1, const float& y1, const float& x2, const float& y2)
            :x1(x1),y1(y1),x2(x2),y2(y2){}

        float   x1      =   0;
        float   y1      =   0;
        float   x2      =   0;
        float   y2      =   0;
    };

    static inline X1Y1X2Y2Box toX1Y1X2Y2Box( const XYWHBox &box)
    {
        float   x1  =   box.x - box.w/2.f;
        float   x2  =   box.x + box.w/2.f;
        float   y1  =   box.y - box.h/2.f;
        float   y2  =   box.y + box.h/2.f;
        return X1Y1X2Y2Box(x1,y1,x2,y2);
    }

    static inline XYWHBox toXYWHBox(const X1Y1X2Y2Box &box)
    {
        float   w   =   box.x2 - box.x1;
        float   h   =   box.y2 - box.y1;
        float   x   =   box.x1 + w/2.f;
        float   y   =   box.y1 + h/2.f;

        return XYWHBox(x,y,w,h);
    }

    static inline float iou(const XYWHBox &box1, const XYWHBox &box2)
    {
        X1Y1X2Y2Box b1      = toX1Y1X2Y2Box(box1);  

        X1Y1X2Y2Box b2      = toX1Y1X2Y2Box(box2);  

        float innerRectX1   = std::max(b1.x1,b2.x1);
        float innerRectY1   = std::max(b1.y1,b2.y1);
        float innerRectX2   = std::min(b1.x2,b2.x2);
        float innerRectY2   = std::min(b1.y2,b2.y2);
        float innerW        = (innerRectX2 - innerRectX1 + 1);
        float innerH        = (innerRectY2 - innerRectY1 + 1);
        float innerRectArea = (innerW<0?0:innerW)*(innerH<0?0:innerH);
        float b1Area        = (b1.x2 - b1.x1 + 1)*(b1.y2 - b1.y1 + 1);
        float b2Area        = (b2.x2 - b2.x1 + 1)*(b2.y2 - b2.y1 + 1);

        return  1.f * innerRectArea / (b1Area + b2Area - innerRectArea);
    }
};

}

#endif 

