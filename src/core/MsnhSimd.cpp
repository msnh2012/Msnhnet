#include "Msnhnet/core/MsnhSimd.h"

namespace Msnhnet
{
#ifdef USE_X86
bool SimdInfo::supportSSE    = false;
bool SimdInfo::supportSSE2   = false;
bool SimdInfo::supportSSE3   = false;
bool SimdInfo::supportSSSE3  = false;
bool SimdInfo::supportSSE4_1 = false;
bool SimdInfo::supportSSE4_2 = false;
bool SimdInfo::supportFMA3   = false;
bool SimdInfo::supportAVX    = false;
bool SimdInfo::supportAVX2   = false;
bool SimdInfo::supportAVX512 = false;
bool SimdInfo::checked       = false;
#endif
}
