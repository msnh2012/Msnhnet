namespace Msnhnet
{

class InnerProductArm
{
public:
    void InnerProduct(float *const &src,  const int &inChannel,  float *const &weight, float* &dest, const int& outChannel);
};

}