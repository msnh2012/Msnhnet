//

namespace Msnhnet
{

class ConvolutionalLayerArm3x3s1
{
public:
    //bottom: src, inw, inh, inch
    //top: dest, outw, outh, outch
    void conv3x3s1_neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel, 
                                        float* &dest, const int &outWidth, const int &outHeight, const int &outChannel);
};

}