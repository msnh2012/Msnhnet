#ifndef MSNHYOLOV3OUTPUTLAYERGPU_H
#define MSNHYOLOV3OUTPUTLAYERGPU_H

#include "Msnhnet/utils/MsnhExport.h"
#include "Msnhnet/config/MsnhnetCuda.h"

namespace Msnhnet
{
class Yolov3OutLayerGPU
{
public:
    static void shuffleData(const int &kn, const int &wxh, const int &chn, float *const &allInput, float *const &shuffled, const int &offset);
};
}

#endif
