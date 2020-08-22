namespace Msnhnet
{

class ConvolutionalLayerArm3x3s2
{
public:
    //bottom: src, inWidth, inHeight, inChannel
    //top: dest, outWidth, outHeight, outChannel
    void conv3x3s2_neon(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float *const &kernel, 
                                        float* &dest, const int &outWidth, const int &outHeight, const int &outChannel);
};

}