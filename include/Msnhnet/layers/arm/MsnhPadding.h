namespace Msnhnet
{

class PaddingLayerArm
{
public:
    //bottom: src, inWidth, inHeight, inChannel
    //top: dest, outWidth, outHeight, outChannel
    void padding(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, 
                                        float* &dest, const int &Top, const int &Down, const int &Left, const int &Right, const int& Val);
};

}