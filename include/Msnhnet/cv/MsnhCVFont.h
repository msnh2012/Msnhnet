#ifndef MSNHCVFONT_H
#define MSNHCVFONT_H

#include <map>
#include <vector>
#include <stdint.h>
#include "Msnhnet/config/MsnhnetCfg.h"

namespace Msnhnet
{

class MsnhNet_API Font
{
public:
    static std::map<uint8_t, std::vector<uint8_t>> fontLib;
    static void init();
    static bool inited;
};

}

#endif

