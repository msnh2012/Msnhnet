#include "Msnhnet/net/MsnhNetwork.h"
namespace Msnhnet
{
NetworkState::~NetworkState()
{
    releaseArr(workspace);

#ifdef USE_GPU
    Cuda::freeCuda(gpuWorkspace);
#endif
}
}
