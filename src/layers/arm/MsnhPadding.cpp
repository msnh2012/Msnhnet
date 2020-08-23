#include "Msnhnet/layers/arm/MsnhPadding.h"
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{
    void PaddingLayerArm::padding(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, 
                                        float* &dest, const int &Top, const int &Down, const int &Left, const int &Right, const int &Val){
        
        int outWidth = inWidth + Left + Right;
        int outHeight = inHeight + Top + Down;
        int i = 0;
        float *destPtr = dest;
        // fill top
        for(; i < Top; i++){
            int j = 0;
            for(; j < outWidth; j++){
                destPtr[j] = Val;
            }
            destPtr += outWidth;
        }

        // fill center
        for(; i < Top + inHeight; i++){
            int j = 0;
            for(; j < Left; j++){
                destPtr[j] = Val;
            }
            
        }
    }
}


