#include <Msnhnet/cv/MsnhCVDraw.h>

namespace Msnhnet
{

void Draw::drawLine(Mat &mat, Vec2I32 p1, Vec2I32 p2, const Vec3U8 &color)
{
    checkMat(mat);

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

    if(std::abs(p2.x1-p1.x1)>std::abs(p2.x2-p1.x2))
    {
        int32_t base = std::min(p1.x1,p2.x1);

        for (int32_t i = 0; i < std::abs(p1.x1 - p2.x1); ++i)
        {
            int32_t x = i + base;

            int32_t y = static_cast<int32_t>(alpha*x + beta);

            if(mat.getChannel()==1)
            {
                uint8_t mColor = static_cast<uint8_t>(color.x1+color.x2+color.x3)/3;
                mat.setPixel<uint8_t>(Vec2I32(x,y),mColor);
            }
            else if(mat.getChannel()==3)
            {
                mat.setPixel<Vec3U8>(Vec2I32(x,y),color);
            }
            else if(mat.getChannel()==4)
            {
                Vec4U8 mColor(color.x1,color.x2,color.x3,255);
                mat.setPixel<Vec4U8>(Vec2I32(x,y),mColor);
            }
        }
    }
    else
    {
        int32_t base = std::min(p1.x2,p2.x2);

        for (int32_t i = 0; i < std::abs(p1.x2 - p2.x2); ++i)
        {

            int32_t y = i + base;

            int32_t x = static_cast<int32_t>((y - beta)/alpha);

            if(mat.getChannel()==1)
            {
                uint8_t mColor = static_cast<uint8_t>(color.x1+color.x2+color.x3)/3;
                mat.setPixel<uint8_t>(Vec2I32(x,y),mColor);
            }
            else if(mat.getChannel()==3)
            {
                mat.setPixel<Vec3U8>(Vec2I32(x,y),color);
            }
            else if(mat.getChannel()==4)
            {
                Vec4U8 mColor(color.x1,color.x2,color.x3,255);
                mat.setPixel<Vec4U8>(Vec2I32(x,y),mColor);
            }
        }
    }

}

void Draw::drawLine(Mat &mat, Vec2I32 p1, Vec2I32 p2, const Vec3U8 &color, const int &width)
{

    if(width <= 1)
    {
        drawLine(mat, p1, p2, color);
    }
    else
    {
        int dx = p1.x1-p2.x1;
        int dy = p1.x2-p2.x2;

        if(dx == 0)
        {
            drawLine(mat, p1, p2, color);
            int w = 1;
            for (int i = 1; i < width; ++i)
            {
                Vec2I32 p11 = p1;
                Vec2I32 p12 = p2;
                if(i%2 == 1)
                {
                    p11.x1 = p11.x1-w;
                    p12.x1 = p12.x1-w;
                    p11.x2 = p11.x2;
                    p12.x2 = p12.x2;
                    drawLine(mat, p11, p12, color);
                }
                else if(i%2 == 0)
                {
                    p11.x1 = p11.x1+w;
                    p12.x1 = p12.x1+w;
                    p11.x2 = p11.x2;
                    p12.x2 = p12.x2;
                    drawLine(mat, p11, p12, color);
                    w++;
                }
            }
        }
        else if(dy == 0)
        {
            drawLine(mat, p1, p2, color);
            int w = 1;
            for (int i = 1; i < width; ++i)
            {
                Vec2I32 p11 = p1;
                Vec2I32 p12 = p2;
                if(i%2 == 1)
                {
                    p11.x1 = p11.x1;
                    p12.x1 = p12.x1;
                    p11.x2 = p11.x2-w;
                    p12.x2 = p12.x2-w;
                    drawLine(mat, p11, p12, color);
                }
                else if(i%2 == 0)
                {
                    p11.x1 = p11.x1;
                    p12.x1 = p12.x1;
                    p11.x2 = p11.x2+w;
                    p12.x2 = p12.x2+w;
                    drawLine(mat, p11, p12, color);
                    w++;
                }
            }
        }
        else if(1.0f*dx/dy<0)
        {
            drawLine(mat, p1, p2, color);

            float fx = 1.f;
            float fy = 1.f ;

            if(std::abs(dx) < std::abs(dy))
            {
                fy = -1.f*dx/dy;
            }

            if(std::abs(dx) > std::abs(dy))
            {
                fx = -1.f*dy/dx;
            }

            int k = 1;
            int w = 1;

            for (int i = 1; i < width; ++i)
            {
                Vec2I32 p11 = p1;
                Vec2I32 p12 = p2;
                if(k == 1)
                {
                    p11.x1 = p11.x1-int(w*fx-1);
                    p12.x1 = p12.x1-int(w*fx);
                    p11.x2 = p11.x2-int(w*fy);
                    p12.x2 = p12.x2-int(w*fy-1);
                    drawLine(mat, p11, p12, color);
                }
                else if(k == 2)
                {
                    p11.x1 = p11.x1+int(w*fx);
                    p12.x1 = p12.x1+int(w*fx-1);
                    p11.x2 = p11.x2+int(w*fy-1);
                    p12.x2 = p12.x2+int(w*fy);
                    drawLine(mat, p11, p12, color);
                }
                else if(k == 3)
                {
                    p11.x1 = p11.x1-int(w*fx);
                    p12.x1 = p12.x1-int(w*fx);
                    p11.x2 = p11.x2-int(w*fy);
                    p12.x2 = p12.x2-int(w*fy);
                    drawLine(mat, p11, p12, color);
                }
                else if(k == 4)
                {
                    p11.x1 = p11.x1+int(w*fx);
                    p12.x1 = p12.x1+int(w*fx);
                    p11.x2 = p11.x2+int(w*fy);
                    p12.x2 = p12.x2+int(w*fy);
                    drawLine(mat, p11, p12, color);
                    k = 1;
                    w ++;
                    continue;
                }
                k++;
            }
        }
        else if(1.0f*dx/dy>0)
        {
            drawLine(mat, p1, p2, color);

            float fx = 1.f;
            float fy = 1.f ;

            if(std::abs(dx) < std::abs(dy))
            {
                fy = 1.f*dx/dy;
            }

            if(std::abs(dx) > std::abs(dy))
            {
                fx = 1.f*dy/dx;
            }

            int k = 1;
            int w = 1;

            for (int i = 1; i < width; ++i)
            {
                Vec2I32 p11 = p1;
                Vec2I32 p12 = p2;
                if(k == 1)
                {
                    p11.x1 = p11.x1+int(w*fx-1);
                    p12.x1 = p12.x1+int(w*fx);
                    p11.x2 = p11.x2-int(w*fy);
                    p12.x2 = p12.x2-int(w*fy-1);
                    drawLine(mat, p11, p12, color);
                }
                else if(k == 2)
                {
                    p11.x1 = p11.x1-int(w*fx);
                    p12.x1 = p12.x1-int(w*fx-1);
                    p11.x2 = p11.x2+int(w*fy-1);
                    p12.x2 = p12.x2+int(w*fy);
                    drawLine(mat, p11, p12, color);
                }
                else if(k == 3)
                {
                    p11.x1 = p11.x1+int(w*fx);
                    p12.x1 = p12.x1+int(w*fx);
                    p11.x2 = p11.x2-int(w*fy);
                    p12.x2 = p12.x2-int(w*fy);
                    drawLine(mat, p11, p12, color);
                }
                else if(k == 4)
                {
                    p11.x1 = p11.x1-int(w*fx);
                    p12.x1 = p12.x1-int(w*fx);
                    p11.x2 = p11.x2+int(w*fy);
                    p12.x2 = p12.x2+int(w*fy);
                    drawLine(mat, p11, p12, color);
                    k = 1;
                    w ++;
                    continue;
                }
                k++;
            }
        }
    }

}

void Draw::drawRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec3U8 &color, const int &width)
{
    checkMat(mat);

    Vec2I32 p3(p2.x1,p1.x2);
    Vec2I32 p4(p1.x1,p2.x2);

    drawLine(mat, p1, p3, color, width);
    drawLine(mat, p2, p3, color, width);
    drawLine(mat, p1, p4, color, width);
    drawLine(mat, p2, p4, color, width);

}

void Draw::drawRect(Mat &mat, const Vec2I32 &pos, const int32_t &width, const int32_t &height, const Vec3U8 &color, const int &lineWidth)
{
    checkMat(mat);

    Vec2I32 p1(pos.x1-width/2,pos.x2-height/2);
    Vec2I32 p2(pos.x1+width/2,pos.x2-height/2);
    Vec2I32 p3(pos.x1+width/2,pos.x2+height/2);
    Vec2I32 p4(pos.x1-width/2,pos.x2+height/2);

    drawLine(mat, p1, p2, color, lineWidth);
    drawLine(mat, p2, p3, color, lineWidth);
    drawLine(mat, p3, p4, color, lineWidth);
    drawLine(mat, p4, p1, color, lineWidth);
}

void Draw::drawRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec2I32 &p3, const Vec2I32 &p4, const Vec3U8 &color, const int &width)
{
    checkMat(mat);

    drawLine(mat, p1, p2, color, width);
    drawLine(mat, p2, p3, color, width);
    drawLine(mat, p3, p4, color, width);
    drawLine(mat, p4, p1, color, width);
}

void Draw::drawPoly(Mat &mat, std::vector<Vec2I32> &points, const Vec3U8 &color, const int &width)
{
    checkMat(mat);

    for (int i = 0; i < points.size()-1; ++i)
    {
        drawLine(mat, points[i], points[i+1], color, width);
    }
    drawLine(mat, points[0], points[points.size()-1], color, width);
}

void Draw::drawEllipse(Mat &mat, const Vec2I32 &pos, const int32_t &width, const int32_t &height, const Vec3U8 &color)
{
    checkMat(mat);

    Vec2I32 p1;
    Vec2I32 p2;
    Vec2I32 p3;
    Vec2I32 p4;

    int32_t matW  = mat.getWidth();
    int32_t matH  = mat.getHeight();

    for (int i = 0; i < width; ++i)
    {
        bool ignoreP1 = false;
        bool ignoreP2 = false;
        bool ignoreP3 = false;
        bool ignoreP4 = false;

        int x    =  i + pos.x1;
        int negX = -i + pos.x1;

        float theta    = acosf(1.f*i/width);

        float ellipseY = height*sinf(theta);

        int y          = static_cast<int>(ellipseY + pos.x2);
        int negY       = static_cast<int>(-ellipseY + pos.x2);

        if(x<0)
        {
            ignoreP1 = true;
            ignoreP2 = true;
            x = 0;
        }

        if(x>=matW)
        {
            ignoreP1 = true;
            ignoreP2 = true;
            x = matW-1;
        }

        if(negX<0)
        {
            ignoreP3 = true;
            ignoreP4 = true;
            negX = 0;
        }

        if(negX>=matW)
        {
            ignoreP3 = true;
            ignoreP4 = true;
            negX = matW-1;
        }

        if(y<0)
        {
            ignoreP1 = true;
            ignoreP3 = true;
            y=0;
        }

        if(y>=matH)
        {
            ignoreP1 = true;
            ignoreP3 = true;
            y = matH-1;
        }

        if(negY<0)
        {
            ignoreP2 = true;
            ignoreP4 = true;
            negY = 0;
        }

        if(negY>=matH)
        {
            ignoreP2 = true;
            ignoreP4 = true;
            negY = matH-1;
        }

        p1.x1 = static_cast<int32_t>(x);
        p2.x1 = static_cast<int32_t>(x);
        p3.x1 = static_cast<int32_t>(negX);
        p4.x1 = static_cast<int32_t>(negX);

        p1.x2 = static_cast<int32_t>(y);
        p2.x2 = static_cast<int32_t>(negY);
        p3.x2 = static_cast<int32_t>(y);
        p4.x2 = static_cast<int32_t>(negY);

        if(mat.getChannel()==1)
        {
            uint8_t mColor = static_cast<uint8_t>(color.x1+color.x2+color.x3)/3;
            if(!ignoreP1)
                mat.setPixel<uint8_t>(p1,mColor);
            if(!ignoreP2)
                mat.setPixel<uint8_t>(p2,mColor);
            if(!ignoreP3)
                mat.setPixel<uint8_t>(p3,mColor);
            if(!ignoreP4)
                mat.setPixel<uint8_t>(p4,mColor);
        }
        else if(mat.getChannel()==3)
        {
            Vec3U8 mColor = color;
            if(!ignoreP1)
                mat.setPixel<Vec3U8>(p1,mColor);
            if(!ignoreP2)
                mat.setPixel<Vec3U8>(p2,mColor);
            if(!ignoreP3)
                mat.setPixel<Vec3U8>(p3,mColor);
            if(!ignoreP4)
                mat.setPixel<Vec3U8>(p4,mColor);
        }
        else if(mat.getChannel()==4)
        {
            Vec4U8 mColor(color.x1,color.x2,color.x3,255);
            if(!ignoreP1)
                mat.setPixel<Vec4U8>(p1,mColor);
            if(!ignoreP2)
                mat.setPixel<Vec4U8>(p2,mColor);
            if(!ignoreP3)
                mat.setPixel<Vec4U8>(p3,mColor);
            if(!ignoreP4)
                mat.setPixel<Vec4U8>(p4,mColor);
        }
    }

    for (int i = 0; i < height; ++i)
    {
        bool ignoreP1 = false;
        bool ignoreP2 = false;
        bool ignoreP3 = false;
        bool ignoreP4 = false;

        int y    =  i + pos.x2;
        int negY = -i + pos.x2;

        float theta    = asinf(1.f*i/height);

        float ellipseX = width*cosf(theta);

        int x          = static_cast<int>(ellipseX + pos.x1);
        int negX       = static_cast<int>(-ellipseX + pos.x1);

        if(x<0)
        {
            ignoreP1 = true;
            ignoreP2 = true;
            x = 0;
        }

        if(x>=matW)
        {
            ignoreP1 = true;
            ignoreP2 = true;
            x = matW-1;
        }

        if(negX<0)
        {
            ignoreP3 = true;
            ignoreP4 = true;
            negX = 0;
        }

        if(negX>=matW)
        {
            ignoreP3 = true;
            ignoreP4 = true;
            negX = matW-1;
        }

        if(y<0)
        {
            ignoreP1 = true;
            ignoreP3 = true;
            y=0;
        }

        if(y>=matH)
        {
            ignoreP1 = true;
            ignoreP3 = true;
            y = matH-1;
        }

        if(negY<0)
        {
            ignoreP2 = true;
            ignoreP4 = true;
            negY = 0;
        }

        if(negY>=matH)
        {
            ignoreP2 = true;
            ignoreP4 = true;
            negY = matH-1;
        }

        p1.x1 = static_cast<int32_t>(x);
        p2.x1 = static_cast<int32_t>(x);
        p3.x1 = static_cast<int32_t>(negX);
        p4.x1 = static_cast<int32_t>(negX);

        p1.x2 = static_cast<int32_t>(y);
        p2.x2 = static_cast<int32_t>(negY);
        p3.x2 = static_cast<int32_t>(y);
        p4.x2 = static_cast<int32_t>(negY);

        if(mat.getChannel()==1)
        {
            uint8_t mColor = static_cast<uint8_t>(color.x1+color.x2+color.x3)/3;
            if(!ignoreP1)
                mat.setPixel<uint8_t>(p1,mColor);
            if(!ignoreP2)
                mat.setPixel<uint8_t>(p2,mColor);
            if(!ignoreP3)
                mat.setPixel<uint8_t>(p3,mColor);
            if(!ignoreP4)
                mat.setPixel<uint8_t>(p4,mColor);
        }
        else if(mat.getChannel()==3)
        {
            Vec3U8 mColor = color;
            if(!ignoreP1)
                mat.setPixel<Vec3U8>(p1,mColor);
            if(!ignoreP2)
                mat.setPixel<Vec3U8>(p2,mColor);
            if(!ignoreP3)
                mat.setPixel<Vec3U8>(p3,mColor);
            if(!ignoreP4)
                mat.setPixel<Vec3U8>(p4,mColor);
        }
        else if(mat.getChannel()==4)
        {
            Vec4U8 mColor(color.x1,color.x2,color.x3,255);
            if(!ignoreP1)
                mat.setPixel<Vec4U8>(p1,mColor);
            if(!ignoreP2)
                mat.setPixel<Vec4U8>(p2,mColor);
            if(!ignoreP3)
                mat.setPixel<Vec4U8>(p3,mColor);
            if(!ignoreP4)
                mat.setPixel<Vec4U8>(p4,mColor);
        }
    }

}

void Draw::drawEllipse(Mat &mat, const Vec2I32 &pos, const int32_t &width, const int32_t &height, const Vec3U8 &color, const int &lineWidth)
{
    checkMat(mat);
    if(lineWidth<=1)
    {
        drawEllipse(mat, pos, width, height, color);
    }
    else
    {
        drawEllipse(mat, pos, width, height, color);
        int k = 1;
        for (int i = 1; i < lineWidth; ++i)
        {
            if(i%2==1)
            {
                drawEllipse(mat, pos, width-k, height-k, color);
            }
            else if(i%2==0)
            {
                drawEllipse(mat, pos, width+k, height+k, color);
                k++;
            }
        }
    }

}

void Draw::fillRect(Mat &mat, const Vec2I32 &p1, const Vec2I32 &p2, const Vec3U8 &color, const bool &addMode)
{
    checkMat(mat);

    int32_t w = std::abs(p2.x1-p1.x1);
    int32_t h = std::abs(p2.x2-p1.x2);

    int32_t baseW = std::min(p1.x1,p2.x1);
    int32_t baseH = std::min(p1.x2,p2.x2);

    int32_t matW  = mat.getWidth();
    int32_t matH  = mat.getHeight();

    for (int32_t i = 0; i < h; ++i)
    {
        for (int32_t j = 0; j < w; ++j)
        {
            int32_t x = j + baseW;
            int32_t y = i + baseH;

            if(mat.getChannel()==1)
            {
                uint8_t mColor = static_cast<uint8_t>(color.x1+color.x2+color.x3)/3;
                if(addMode)
                {
                    uint8_t tmpPixel = mat.getPixel<uint8_t>(Vec2I32(x,y));
                    uint16_t addVal  = tmpPixel + mColor;
                    if(addVal>255)
                    {
                        mColor = 255;
                    }
                    else
                    {
                        mColor = static_cast<uint8_t>(addVal);
                    }
                }
                mat.setPixel<uint8_t>(Vec2I32(x,y),mColor);
            }
            else if(mat.getChannel()==3)
            {
                Vec3U8 mColor = color;
                if(addMode)
                {
                    Vec3U8 tmpPixel = mat.getPixel<Vec3U8>(Vec2I32(x,y));
                    Vec3U16 addVal;

                    addVal.x1 = tmpPixel.x1 + color.x1;
                    addVal.x2 = tmpPixel.x2 + color.x2;
                    addVal.x3 = tmpPixel.x3 + color.x3;

                    mColor.x1 = (addVal.x1>255)?255:static_cast<uint8_t>(addVal.x1);
                    mColor.x2 = (addVal.x2>255)?255:static_cast<uint8_t>(addVal.x2);
                    mColor.x3 = (addVal.x3>255)?255:static_cast<uint8_t>(addVal.x3);
                }
                mat.setPixel<Vec3U8>(Vec2I32(x,y),mColor);
            }
            else if(mat.getChannel()==4)
            {
                Vec4U8 mColor(color.x1,color.x2,color.x3,255);
                if(addMode)
                {
                    Vec4U8 tmpPixel = mat.getPixel<Vec4U8>(Vec2I32(x,y));
                    Vec3U16 addVal;

                    addVal.x1 = tmpPixel.x1 + color.x1;
                    addVal.x2 = tmpPixel.x2 + color.x2;
                    addVal.x3 = tmpPixel.x3 + color.x3;

                    mColor.x1 = (addVal.x1>255)?255:static_cast<uint8_t>(addVal.x1);
                    mColor.x2 = (addVal.x2>255)?255:static_cast<uint8_t>(addVal.x2);
                    mColor.x3 = (addVal.x3>255)?255:static_cast<uint8_t>(addVal.x3);
                }
                mat.setPixel<Vec4U8>(Vec2I32(x,y),mColor);
            }
        }
    }

}

void Draw::fillEllipse(Mat &mat, const Vec2I32 &pos, const int32_t &width, const int32_t &height, const Vec3U8 &color)
{
    checkMat(mat);

    Vec2I32 p1;
    Vec2I32 p2;
    Vec2I32 p3;
    Vec2I32 p4;

    int32_t matW  = mat.getWidth();
    int32_t matH  = mat.getHeight();

    for (int i = 0; i < width; ++i)
    {

        int x    =  i + pos.x1;
        int negX = -i + pos.x1;

        float theta    = acosf(1.f*i/width);

        float ellipseY = height*sinf(theta);

        int y          = static_cast<int>(ellipseY + pos.x2);
        int negY       = static_cast<int>(-ellipseY + pos.x2);

        if(x<0)
        {
            x = 0;
        }

        if(x>=matW)
        {
            x = matW-1;
        }

        if(negX<0)
        {
            negX = 0;
        }

        if(negX>=matW)
        {
            negX = matW-1;
        }

        if(y<0)
        {
            y=0;
        }

        if(y>=matH)
        {
            y = matH-1;
        }

        if(negY<0)
        {
            negY = 0;
        }

        if(negY>=matH)
        {
            negY = matH-1;
        }

        p1.x1 = x;
        p2.x1 = x;
        p3.x1 = negX;
        p4.x1 = negX;

        p1.x2 = y;
        p2.x2 = negY;
        p3.x2 = y;
        p4.x2 = negY;

        drawLine(mat, p1, p2, color);
        drawLine(mat, p3, p4, color);

    }
}

void Draw::drawFont(Mat &mat, const std::string &content, const Vec2I32 &pos, const Vec3U8 &color)
{
    checkMat(mat);

    Font::init();

    for (int i = 0; i < content.length(); ++i)
    {
        uint8_t ch = content[i];
        auto it = Font::fontLib.find(ch);

        if(it==Font::fontLib.end())
        {
            throw Exception(1, "[CV]: contains unsupported char!",__FILE__,__LINE__,__FUNCTION__);
        }

        std::vector<uint8_t> font = Font::fontLib[ch];

        for (int m = 0; m < 32; ++m)
        {
            for (int n = 0; n < 16; ++n)
            {
                uint8_t mask = font[m*16+n];

                if(mask==0)
                    continue;
                else
                {
                    if((pos.x1+i*16+n) >= mat.getWidth())
                    {
                        continue;
                    }

                    if((pos.x2+m) >= mat.getHeight())
                    {
                        continue;
                    }
                }

                Vec2I32 pix;

                pix.x1 = pos.x1 + i*16 + n;
                pix.x2 = pos.x2 + m;

                if(mat.getChannel()==1)
                {
                    uint8_t mColor = static_cast<uint8_t>(color.x1+color.x2+color.x3)/3;
                    mat.setPixel<uint8_t>(pix,mColor);
                }
                else if(mat.getChannel()==3)
                {
                    Vec3U8 mColor = color;
                    mat.setPixel<Vec3U8>(pix,mColor);
                }
                else if(mat.getChannel()==4)
                {
                    Vec4U8 mColor(color.x1,color.x2,color.x3,255);
                    mat.setPixel<Vec4U8>(pix,mColor);
                }
            }
        }

    }
}

void Draw::checkMat(Mat &mat)
{
    MatType srcMatType = mat.getMatType();

    if(srcMatType != MatType::MAT_GRAY_U8 && srcMatType != MatType::MAT_RGB_U8 && srcMatType != MatType::MAT_RGBA_U8)
    {
        throw Exception(1,"[CV]: draw functions only support for uint8_t mat!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(mat.isEmpty())
    {
        throw Exception(1,"[CV]: mat Empty!", __FILE__, __LINE__, __FUNCTION__);
    }
}
}
