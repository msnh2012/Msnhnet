#ifndef MSNHCVFONT_H
#define MSNHCVFONT_H

#include <map>
#include <vector>
#include <stdint.h>

namespace Msnhnet
{

class Font
{
public:
    static std::map<uint8_t, std::vector<uint8_t>> fontLib;
    static void init();
    static bool inited;
};

}

#endif

