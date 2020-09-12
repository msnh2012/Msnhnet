#include "Msnhnet/layers/arm/MsnhPadding.h"
#include "Msnhnet/layers/arm/MsnhMaxPooling2x2.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{


//bottom: src, inWidth, inHeight, inChannel
//top: dest, outWidth, outHeight, outChannel
void MaxPooling2x2Arm::pooling(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, 
                                    float* &dest, const int& outWidth, const int& outHeight, const int& outChannel){
        
    }


}