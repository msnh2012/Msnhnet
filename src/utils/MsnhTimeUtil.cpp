#include "Msnhnet/utils/MsnhTimeUtil.h"

namespace Msnhnet
{
std::chrono::time_point<std::chrono::high_resolution_clock> TimeUtil::st;
std::chrono::time_point<std::chrono::high_resolution_clock> TimeUtil::so;

TimeUtil::TimeUtil()
{
    std::chrono::high_resolution_clock::now();
}

void TimeUtil::startRecord()
{
    st = std::chrono::high_resolution_clock::now();
}

float TimeUtil::getElapsedTime()
{
    so = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float,std::milli>(so - st).count();
}

}
