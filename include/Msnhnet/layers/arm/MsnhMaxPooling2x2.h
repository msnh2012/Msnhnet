namespace Msnhnet
{

class MaxPooling2x2Arm
{
public:
    //bottom: src, inWidth, inHeight, inChannel
    //top: dest, outWidth, outHeight, outChannel
    void pooling(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, 
                                        float* &dest, const int& outWidth, const int& outHeight, const int& outChannel);
};

}