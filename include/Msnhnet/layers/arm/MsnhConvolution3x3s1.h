//

namespace Msnhnet
{

class ConvolutionalLayerArm3x3s1
{
public:
    //bottom: src, inw, inh, inch
    //top: dest, outw, outh, outch
    void conv3x3s1_neon(float *const &src, const int &inw, const int &inh,  const int &inch, float *const &kernel, const int &kw, 
                        const int &kh, float* &dest, const int &outw, const int &outh, const int &outch);
};

}