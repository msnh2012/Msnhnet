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
#include <string.h>
#include <math.h>
#endif
#endif

#ifdef USE_ARM
#include <math.h>
#include <string.h>
#include <arm_neon.h>
#endif

namespace Msnhnet
{
using namespace std;
class SimdInfo
{
public:
    SimdInfo()
    {

    }

    ~SimdInfo(){}

#ifdef USE_X86
    bool getSupportSSE() const
    {
        return supportSSE;
    }

    bool getSupportSSE2() const
    {
        return supportSSE2;
    }

    bool getSupportSSE3() const
    {
        return supportSSE3;
    }

    bool getSupportSSSE3() const
    {
        return supportSSSE3;
    }

    bool getSupportSSE4_1() const
    {
        return supportSSE4_1;
    }

    bool getSupportSSE4_2() const
    {
        return supportSSE4_2;
    }

    bool getSupportAVX() const
    {
        return supportAVX;
    }

    bool getSupportAVX2() const
    {
        return supportAVX2;
    }

    bool getSupportAVX512() const
    {
        return supportAVX512;
    }

    bool getSupportFMA3() const
    {
        return supportFMA3;
    }

    bool checkSimd()
    {
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
        return true;
#endif
    }

private:
    bool supportSSE    = false;
    bool supportSSE2   = false;
    bool supportSSE3   = false;
    bool supportSSSE3  = false;
    bool supportSSE4_1 = false;
    bool supportSSE4_2 = false;
    bool supportFMA3   = false;
    bool supportAVX    = false;
    bool supportAVX2   = false;
    bool supportAVX512 = false;

#ifdef WIN32
    inline std::array<unsigned int,4> cpuid(int function_id)
    {
        std::array<unsigned int,4> info;

        __cpuid((int*)info.data(), function_id);
        return info;
    }
    inline bool cpuHasSSE()     {return 0!=(cpuid(1)[3]&(1<<25));}
    inline bool cpuHasSSE2()    { return 0!=(cpuid(1)[3]&(1<<26)); }
    inline bool cpuHasSSE3()    { return 0!=(cpuid(1)[2]&(1<<0));  }
    inline bool cpuHasSSSE3()   { return 0!=(cpuid(1)[2]&(1<<9));  }
    inline bool cpuHasSSE4_1()  { return 0!=(cpuid(1)[2]&(1<<19)); }
    inline bool cpuHasSSE4_2()  { return 0!=(cpuid(1)[2]&(1<<20)); }
    inline bool cpuHasFMA3()    { return 0!=(cpuid(1)[2]&(1<<12)); }

    inline bool cpuHasAVX()     { return 0!=(cpuid(1)[2]&(1<<28)); }
    inline bool cpuHasAVX2()    { return 0!=(cpuid(7)[1]&(1<<5));  }
    inline bool cpuHasAVX512()  { return 0!=(cpuid(7)[1]&(1<<16)); }
#endif
#endif

};

}
#endif 
