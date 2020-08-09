#include "Msnhnet/net/MsnhNetwork.h"
namespace Msnhnet
{
NetworkState::~NetworkState()
{
    releaseArr(workspace);

#ifdef USE_GPU
    Cuda::freeCuda(gpuWorkspace);
    if(BaseLayer::useFp16)
    {
        Cuda::freeCuda(gpuInputFp16);
    }
#endif
}
}
