//

namespace Msnhnet
{

class ConvolutionalLayerArm3x3
{
public:
    //bottom: src, inw, inh, inch
    //top: dest, outw, outh, outch
    void conv3x3s1_neon(float *src, int inw, int inh,  int inch, float *kernel, int kw, int kh, float *dest, int ouw, int outw, int outch);
    void conv3x3s2_neon(float *src, int inw, int inh,  int inch, float *kernel, int kw, int kh, float *dest, int ouw, int outw, int outch);
    void conv3x3s1_neon_int8(float *src, int inw, int inh,  int inch, float *kernel, int kw, int kh, float *dest, int ouw, int outw, int outch);
    void conv3x3s2_neon_int8(float *src, int inw, int inh,  int inch, float *kernel, int kw, int kh, float *dest, int ouw, int outw, int outch);
};

}