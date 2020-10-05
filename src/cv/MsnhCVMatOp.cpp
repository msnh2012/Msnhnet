#include "Msnhnet/cv/MsnhCVMatOp.h"
namespace Msnhnet
{

void MatOp::roi(const Mat &src, Mat &dst, const Vec2I32 &p1, const Vec2I32 &p2)
{

    int32_t width   = abs(p1.x1 - p2.x1);
    int32_t height  = abs(p1.x2 - p2.x2);
    int channel     = src.getChannel();
    MatType matType = src.getMatType();
    int step        = src.getStep();

    uint8_t* u8Ptr =  new uint8_t[dst.getWidth()*dst.getHeight()*dst.getStep()]();

    if(p1.x1 < 0 || p1.x2 < 0 || p1.x1 >= src.getWidth() || p1.x2>= src.getHeight() ||
       p2.x1 < 0 || p2.x2 < 0 || p2.x1 >= src.getWidth() || p2.x2>= src.getHeight()
      )
    {
        throw Exception(1,"[CV]: roi point pos out of memory", __FILE__, __LINE__, __FUNCTION__);
    }

    for (int i = 0; i < height; ++i)
    {
        memcpy(u8Ptr+i*width*dst.getStep(), src.getData().u8 + (p1.x2+i)*src.getWidth()*src.getStep() + p1.x1*src.getStep(),width*src.getStep());
    }

    dst.clearMat();
    dst.setChannel(channel);
    dst.setMatType(matType);
    dst.setStep(step);
    dst.setWidth(width);
    dst.setHeight(height);
    dst.setU8Ptr(u8Ptr);
}

}
