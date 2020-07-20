#include "Msnhnet/utils/MsnhMathUtils.h"
namespace Msnhnet
{
float MathUtils::sumArray(float * const &x, const int &xNum)
{
    float sum = 0;
    for (int i = 0; i < xNum; ++i)
    {
        sum = sum + x[i];
    }
    return sum;
}

float MathUtils::meanArray(float * const &x, const int &xNum)
{
    return sumArray(x,xNum)/xNum;
}

float MathUtils::randUniform(float min, float max)
{
    if(max < min)
    {
        float swap = min;
        min = max;
        max = swap;
    }

#if (RAND_MAX < 65536)
    int rnd = rand()*(RAND_MAX + 1) + rand();
    return ((float)rnd / (RAND_MAX*RAND_MAX) * (max - min)) + min;
#else
    return ((float)rand() / RAND_MAX * (max - min)) + min;
#endif
}

float MathUtils::randNorm()
{
    static int haveSapre    = 0;
    static double  rand1    = 0;
    static double  rand2    = 0;

    if(haveSapre)
    {
        haveSapre = 0;
        return static_cast<float>(sqrt(rand1) * sin(rand2));
    }

    haveSapre   = 1;

    rand1       =   randomGen() / ((double)RAND_MAX);
    if(rand1 < 1e-100)
    {
        rand1   =  1e-100;
    }
    rand1       =   -2 * log(rand1);
    rand2       =   (randomGen() / ((double)RAND_MAX)) * 2.0 *M_PI;

    return static_cast<float>(sqrt(rand1) * cos(rand2));

}

unsigned int MathUtils::randomGen()
{
    unsigned int rnd = 0;
    rnd = rand();
#if (RAND_MAX < 65536)
    rnd = rand()*(RAND_MAX + 1) + rnd;
#endif  
    return rnd;
}
}
