#ifndef MSNHPADDINGLAYERGPU_H
#define MSNHPADDINGLAYERGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{

class MsnhNet_API PaddingLayerGPU
{
public:
    static void forwardNormalGPU(const int &batch, const int &outChannel, const int &outHeight, const int &outWidth,
                                 const int &height, const int &width, const int &channel,
                                 const int &top, const int &left,
                                 const float &paddingVal,
                                 float *const &input, float *const &output);
};

}

#endif
