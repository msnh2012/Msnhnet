#define eps 1e-4
namespace Msnhnet
{

class BatchNormLayerArm
{
public:
    void BatchNorm(float * &src, const int &inWidth, const int &inHeight,  const int &inChannel, 
                    float *const &Scales, float *const &rollMean, float *const &rollVariance, float *const &Biases);
};

}