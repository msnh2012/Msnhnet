#define eps 1e-4
namespace Msnhnet
{

class BatchNormLayerArm
{
public:
    void BatchNorm(float *const &src, const int &inWidth, const int &inHeight,  const int &inChannel, float* &dest,
                    float *const &Scales, float *const &rollMean, float *const &rollVariance, float *const &biases);
};

}