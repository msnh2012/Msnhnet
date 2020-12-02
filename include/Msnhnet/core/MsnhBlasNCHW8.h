#ifndef MSNHBLASNCHW8_H
#define MSNHBLASNCHW8_H
#include <algorithm>
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/core/MsnhSimd.h"

namespace Msnhnet
{
#define NCHW8_PACK 8
class MsnhNet_API BlasNCHW8
{
public:
    static void cpuNCHWToNCHW8(float *const &org, const int width, const int height, const int channel, const int batch, float * const &dst);

    static size_t getNCHW8Params(const int width, const int height, const int channel, int &outWidth, int& outChannel);
};

}
#endif 

