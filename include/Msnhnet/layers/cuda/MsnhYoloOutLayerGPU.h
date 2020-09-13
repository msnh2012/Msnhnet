#ifndef MSNHYOLOOUTPUTLAYERGPU_H
#define MSNHYOLOOUTPUTLAYERGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{
class MsnhNet_API YoloOutLayerGPU
{
public:
    static void shuffleData(const int &kn, const int &wxh, const int &chn, float *const &allInput, float *const &shuffled, const int &offset);
};
}

#endif
