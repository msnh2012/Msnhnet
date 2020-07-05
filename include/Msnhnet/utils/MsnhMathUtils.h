#ifndef MSNHMATHUTILS_H
#define MSNHMATHUTILS_H
#include "Msnhnet/config/MsnhnetCfg.h"
#include "Msnhnet/utils/MsnhExport.h"
#include <math.h>
#ifndef M_PI
#define M_PI       3.14159265358979323846   

#endif

namespace Msnhnet
{
class MsnhNet_API MathUtils
{
public:
    static float sumArray(float *const &x, const int &xNum);
    static float meanArray(float *const &x, const int &xNum );

   static float randUniform(float min, float max);

   static float randNorm();

   static unsigned int randomGen();

   template<typename T>
    static inline void swap(T &a, T &b)
    {
        T tmp;
        tmp = a;
        a   = b;
        b   = tmp;
    }
};
}

#endif 

