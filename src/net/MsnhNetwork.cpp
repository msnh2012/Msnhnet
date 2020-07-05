#include "Msnhnet/net/MsnhNetwork.h"
namespace Msnhnet
{
NetworkState::~NetworkState()
{
    releaseArr(workspace);
}
}
