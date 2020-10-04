#include <Msnhnet/cv/MsnhCVDraw.h>

namespace Msnhnet
{

void Draw::drawLine(Mat &mat, Vec2I32 p1, Vec2I32 p2, const Vec3U8 &color)
{
    if(mat.getWidth() == 0 || mat.getHeight() == 0)
    {
        throw Exception(1,"[CV]: img width == 0 || height == 0!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(p1.x1 >= mat.getWidth()) p1.x1 = mat.getWidth()-1;
    if(p1.x1 < 0) p1.x1 = 0;
    if(p2.x1 >= mat.getWidth()) p2.x1 = mat.getWidth()-1;
    if(p2.x1 < 0) p2.x1 = 0;

    if(p1.x2 >= mat.getHeight()) p1.x2 = mat.getWidth()-1;
    if(p1.x2 < 0) p1.x2 = 0;
    if(p2.x2 >= mat.getHeight()) p2.x2 = mat.getWidth()-1;
    if(p2.x2 < 0) p2.x2 = 0;

    float alpha = 0;
    if(p2.x1 == p1.x1)
    {
        alpha = 10000.f;
    }
    else
    {
        alpha = (1.0f*p2.x2 - p1.x2)/(p2.x1 - p1.x1);
    }

    float beta  = p1.x2 - alpha*p1.x1;

    if(abs(p2.x1-p1.x1)>abs(p2.x2-p1.x2))
    {
        uint32_t base = std::min(p1.x1,p2.x1);

        for (uint32_t i = 0; i <abs(p1.x1 - p2.x1); ++i)
        {
            uint32_t x = i + base;

            uint32_t y = static_cast<uint32_t>(alpha*x + beta);

            if(mat.getChannel()==1)
            {
                uint8_t mColor = static_cast<uint8_t>(color.x1+color.x2+color.x3)/3;
                mat.setPixel<uint8_t>(Vec2U32(x,y),mColor);
            }
            else if(mat.getChannel()==3)
            {
                mat.setPixel<Vec3U8>(Vec2U32(x,y),color);
            }
            else if(mat.getChannel()==4)
            {
                Vec4U8 mColor(color.x1,color.x2,color.x3,255);
                mat.setPixel<Vec4U8>(Vec2U32(x,y),mColor);
            }
        }
    }
    else
    {
        uint32_t base = std::min(p1.x2,p2.x2);

        for (uint32_t i = 0; i <abs(p1.x2 - p2.x2); ++i)
        {

            uint32_t y = i + base;

            uint32_t x = static_cast<uint32_t>((y - beta)/alpha);

            if(mat.getChannel()==1)
            {
                uint8_t mColor = static_cast<uint8_t>(color.x1+color.x2+color.x3)/3;
                mat.setPixel<uint8_t>(Vec2U32(x,y),mColor);
            }
            else if(mat.getChannel()==3)
            {
                mat.setPixel<Vec3U8>(Vec2U32(x,y),color);
            }
            else if(mat.getChannel()==4)
            {
                Vec4U8 mColor(color.x1,color.x2,color.x3,255);
                mat.setPixel<Vec4U8>(Vec2U32(x,y),mColor);
            }
        }
    }

}

void Draw::drawRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec3U8 &color)
{
    Vec2I32 p3(p2.x1,p1.x2);
    Vec2I32 p4(p1.x1,p2.x2);

    drawLine(mat, p1, p3, color);
    drawLine(mat, p2, p3, color);
    drawLine(mat, p1, p4, color);
    drawLine(mat, p2, p4, color);

}

void Draw::fillRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec3U8 &color)
{
    uint32_t w = abs(p2.x1-p1.x1);
    uint32_t h = abs(p2.x2-p1.x2);

    uint32_t baseW = std::min(p1.x1,p2.x1);
    uint32_t baseH = std::min(p1.x2,p2.x2);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            uint32_t x = i+baseW;
            uint32_t y = j+baseH;

            x = (x<0)?0:x;
            x = (x>=mat.getWidth())?(mat.getWidth()-1):x;

            y = (y<0)?0:y;
            y = (y>=mat.getHeight())?(mat.getHeight()-1):y;

            if(mat.getChannel()==1)
            {
                uint8_t mColor = static_cast<uint8_t>(color.x1+color.x2+color.x3)/3;
                mat.setPixel<uint8_t>(Vec2U32(x,y),mColor);
            }
            else if(mat.getChannel()==3)
            {
                mat.setPixel<Vec3U8>(Vec2U32(x,y),color);
            }
            else if(mat.getChannel()==4)
            {
                Vec4U8 mColor(color.x1,color.x2,color.x3,255);
                mat.setPixel<Vec4U8>(Vec2U32(x,y),mColor);
            }
        }
    }

}

void Draw::drawRect(Mat &mat, const Vec2I32 &pos, const uint32_t &width, const uint32_t &height, const Vec3U8 &color)
{
    Vec2I32 p1(pos.x1-width/2,pos.x2-height/2);
    Vec2I32 p2(pos.x1+width/2,pos.x2-height/2);
    Vec2I32 p3(pos.x1+width/2,pos.x2+height/2);
    Vec2I32 p4(pos.x1-width/2,pos.x2+height/2);

    drawLine(mat, p1, p2, color);
    drawLine(mat, p2, p3, color);
    drawLine(mat, p3, p4, color);
    drawLine(mat, p4, p1, color);
}

void Draw::drawRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec2I32 &p3, const Vec2I32 &p4, const Vec3U8 &color)
{
    drawLine(mat, p1, p2, color);
    drawLine(mat, p2, p3, color);
    drawLine(mat, p3, p4, color);
    drawLine(mat, p4, p1, color);
}

}
