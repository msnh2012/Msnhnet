#ifndef MSNHSIMD_H
#define MSNHSIMD_H
#include <iostream>
#include <string>
#include "Msnhnet/config/MsnhnetCfg.h"

#ifdef USE_X86
#ifdef WIN32
#include <intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <array>
#else
#include <x86intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <avxintrin.h>
#include <string.h>
#include <math.h>
#endif
#endif

#ifdef USE_ARM
#include <math.h>
#include <string.h>
#ifdef USE_NEON
#include <arm_neon.h>
#endif
#endif

namespace Msnhnet
{
using namespace std;
class SimdInfo
{
public:
#ifdef USE_X86
    static bool checkSimd()
    {
        if(checked)
            return true;
#ifdef linux
        char buf[10240] = {0};
        FILE *pf = NULL;

        string strCmd = "cat /proc/cpuinfo | grep flag";

        if( (pf = popen(strCmd.c_str(), "r")) == NULL )
        {
            return false;
        }

        string strResult;
        while(fgets(buf, sizeof buf, pf))
        {
            strResult += buf;
        }

        pclose(pf);

        unsigned int iSize =  strResult.size();
        if(iSize > 0 && strResult[iSize - 1] == '\n')  

        {
            strResult = strResult.substr(0, iSize - 1);
        }

        if(strResult.find("sse") != string::npos)
        {
            supportSSE = true;
        }

        if(strResult.find("sse2") != string::npos)
        {
            supportSSE2 = true;
        }

        if(strResult.find("sse3") != string::npos)
        {
            supportSSE3 = true;
        }

        if(strResult.find("ssse3") != string::npos)
        {
            supportSSSE3 = true;
        }

        if(strResult.find("sse4_1") != string::npos)
        {
            supportSSE4_1 = true;
        }

        if(strResult.find("sse4_2") != string::npos)
        {
            supportSSE4_2 = true;
        }

        if(strResult.find("avx") != string::npos)
        {
            supportAVX = true;
        }

        if(strResult.find("avx2") != string::npos)
        {
            supportAVX2 = true;
        }

        if(strResult.find("fma") != string::npos)
        {
            supportFMA3 = true;
        }

        if(strResult.find("avx512") != string::npos)
        {
            supportAVX2 = true;
        }
        checked         = true;
        return true;
#endif

#ifdef WIN32
        supportSSE      = cpuHasSSE();
        supportSSE2     = cpuHasSSE2();
        supportSSE3     = cpuHasSSE3();
        supportSSSE3    = cpuHasSSSE3();
        supportSSE4_1   = cpuHasSSE4_1();
        supportSSE4_2   = cpuHasSSE4_2();
        supportFMA3     = cpuHasFMA3();
        supportAVX      = cpuHasAVX();
        supportAVX2     = cpuHasAVX2();
        checked         = true;
        return true;
#endif
    }
    static bool supportSSE   ;
    static bool supportSSE2  ;
    static bool supportSSE3  ;
    static bool supportSSSE3 ;
    static bool supportSSE4_1;
    static bool supportSSE4_2;
    static bool supportFMA3  ;
    static bool supportAVX   ;
    static bool supportAVX2  ;
    static bool supportAVX512;
    static bool checked;
#ifdef WIN32
    static inline std::array<unsigned int,4> cpuid(int function_id)
    {
        std::array<unsigned int,4> info;

        __cpuid((int*)info.data(), function_id);
        return info;
    }
    static inline bool cpuHasSSE()     {return 0!=(cpuid(1)[3]&(1<<25));}
    static inline bool cpuHasSSE2()    { return 0!=(cpuid(1)[3]&(1<<26)); }
    static inline bool cpuHasSSE3()    { return 0!=(cpuid(1)[2]&(1<<0));  }
    static inline bool cpuHasSSSE3()   { return 0!=(cpuid(1)[2]&(1<<9));  }
    static inline bool cpuHasSSE4_1()  { return 0!=(cpuid(1)[2]&(1<<19)); }
    static inline bool cpuHasSSE4_2()  { return 0!=(cpuid(1)[2]&(1<<20)); }
    static inline bool cpuHasFMA3()    { return 0!=(cpuid(1)[2]&(1<<12)); }

    static inline bool cpuHasAVX()     { return 0!=(cpuid(1)[2]&(1<<28)); }
    static inline bool cpuHasAVX2()    { return 0!=(cpuid(7)[1]&(1<<5));  }
    static inline bool cpuHasAVX512()  { return 0!=(cpuid(7)[1]&(1<<16)); }
#endif
#endif

};

}
#endif 

