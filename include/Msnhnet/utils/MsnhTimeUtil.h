#ifndef MSNHTIMEUTIL_H
#define MSNHTIMEUTIL_H

#include <chrono>
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{

class MsnhNet_API TimeUtil
{
public:
    TimeUtil();

    static std::chrono::time_point<std::chrono::high_resolution_clock> startRecord();
    static float getElapsedTime(std::chrono::time_point<std::chrono::high_resolution_clock> &st);
};

}

#endif 

