#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{

class ActivationLayerArm
{
public:
    
    void ActivationLayer(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel, const ActivationType &actType, const float &params);
    void loggyActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void logisticActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void reluActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void relu6Activate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void eluActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void seluActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void relieActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void rampActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void leakyActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel, const float &params);
    void tanhActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void plseActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void stairActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void hardtanActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void lhtanActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void softplusActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel, const float &params);
    void mishActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
    void swishActivate(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel);
};


}