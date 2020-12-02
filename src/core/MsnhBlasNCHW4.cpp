#include "Msnhnet/core/MsnhBlasNCHW4.h"

namespace Msnhnet
{

void BlasNCHW4::cpuNCHWToNCHW4(float * const &org, const int width, const int height, const int channel, const int batch, float * const &dst)
{
    int outWidth   = 0;
    int outHeight  = height;
    int outChannel = 0;
    getNCHW4Params(width, height, channel,outWidth, outChannel);
    for (int b = 0; b < batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int oc = 0; oc < outChannel; ++oc)
        {
            for (int oh = 0; oh < outHeight; ++oh)
            {
                for (int ow = 0; ow < outWidth; ++ow)
                {

                    if(ow%NCHW4_PACK > (channel%NCHW4_PACK-1) && channel%NCHW4_PACK != 0 && (oc+1)*NCHW4_PACK > channel)
                    {
                        dst[b*outChannel*outHeight*outWidth + oc*outHeight*outWidth + oh*outWidth + ow] = 0;
                    }
                    else
                    {
                        dst[b*outChannel*outHeight*outWidth + oc*outHeight*outWidth + oh*outWidth + ow] = org[b*channel*height*width + oc*NCHW4_PACK*width*height + (ow%NCHW4_PACK)*width*height + ow/NCHW4_PACK +  oh*width];
                    }
                }
            }
        }
    }
}

size_t BlasNCHW4::getNCHW4Params(const int width, const int height, const int channel, int &outWidth, int &outChannel)
{
    outChannel = ((channel%NCHW4_PACK) == 0)?channel/NCHW4_PACK:(channel/NCHW4_PACK+1);
    outWidth   = width*NCHW4_PACK;
    return outWidth*height*outChannel;
}

}
