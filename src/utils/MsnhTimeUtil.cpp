#include "Msnhnet/utils/MsnhTimeUtil.h"

namespace Msnhnet
{

TimeUtil::TimeUtil()
{

}

std::chrono::time_point<std::chrono::high_resolution_clock> TimeUtil::startRecord()
{
    return std::chrono::high_resolution_clock::now();
}

float TimeUtil::getElapsedTime(std::chrono::time_point<std::chrono::high_resolution_clock> &st)
{
    auto so = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float,std::milli>(so - st).count();
}

}
