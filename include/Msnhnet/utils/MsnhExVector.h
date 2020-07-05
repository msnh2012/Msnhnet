#ifndef MSNHEXVECTOR_H
#define MSNHEXVECTOR_H
#include <iostream>
#include <algorithm>
#include <vector>
#include <iterator>
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API ExVector
{
public:
    template<typename T>
    static inline bool contains(const std::vector<T>&vec, const T &val)
    {
        auto result = std::find(vec.begin(),vec.end(),val);
        if(result != vec.end())
        {
            return true;
        }
        else
        {
            return false;
        }
    }

   template<typename T>
    static inline long long maxIndex(const std::vector<T>&vec)
    {
        auto biggest = std::max_element(vec.begin(), vec.end());
        return  std::distance(vec.begin(), biggest);
    }

   template<typename T>
    static inline int minIndex(const std::vector<T>&vec)
    {
        auto smallest = std::min_element(vec.begin(), vec.end());
        return  std::distance(vec.begin(), smallest);
    }

   template<typename T>
    static inline T max(const std::vector<T>&vec)
    {
        auto biggest = std::max_element(vec.begin(), vec.end());
        return  *biggest;
    }

   template<typename T>
    static inline T min(const std::vector<T>&vec)
    {
        auto smallest  = std::min_element(vec.begin(), vec.end());
        return  *smallest ;
    }

   template<typename T>
    static inline std::vector<int> argsort(const std::vector<T>& a, const bool &descending = false)
    {
        int Len = a.size();

       std::vector<int> idx(static_cast<size_t>(Len), 0);

       for(int i = 0; i < Len; i++)
        {
            idx[static_cast<size_t>(i)] = i;
        }
        if(!descending)
            std::sort(idx.begin(), idx.end(), [&a](int i1, int i2){return a[i1] < a[i2];});
        else
            std::sort(idx.begin(), idx.end(), [&a](int i1, int i2){return a[i1] > a[i2];});

       return idx;
    }

};
}

#endif 

