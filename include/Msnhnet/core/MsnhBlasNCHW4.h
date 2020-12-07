#ifndef MSNHBLASCHW4_H
#define MSNHBLASCHW4_H
#include <algorithm>
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhSimd.h"

namespace Msnhnet
{
#define NCHW4_PACK 4
class MsnhNet_API BlasNCHW4
{
public:
    static void cpuNCHWToNCHW4(float *const &org, const int width, const int height, const int channel, const int batch, float * const &dst);

    static size_t getNCHW4Params(const int width, const int height, const int channel, int &outWidth, int& outChannel);
};

}

#endif 

