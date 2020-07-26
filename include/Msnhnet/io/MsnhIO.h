#ifndef MSNHIO_H
#define MSNHIO_H
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include "Msnhnet/utils/MsnhExString.h"
#include "Msnhnet/utils/MsnhExport.h"

namespace Msnhnet
{
class MsnhNet_API IO
{
public:

    template<typename T>
    inline static void printVector(std::vector<T>& v,bool needASCII = false)
    {
        if(std::is_same<T,char>::value||
           std::is_same<T,unsigned char>::value||
           std::is_same<T,int>::value||
           std::is_same<T,unsigned int>::value||
           std::is_same<T,float>::value||
           std::is_same<T,double>::value||
           std::is_same<T,long>::value||
           std::is_same<T,unsigned long>::value||
           std::is_same<T,long long>::value||
           std::is_same<T,unsigned long long>::value||
           std::is_same<T,short>::value||
           std::is_same<T,unsigned short>::value||
           std::is_same<T,int8_t>::value||
           std::is_same<T,uint8_t>::value||
           std::is_same<T,int16_t>::value||
           std::is_same<T,uint16_t>::value||
           std::is_same<T,int32_t>::value||
           std::is_same<T,uint32_t>::value||
           std::is_same<T,int64_t>::value||
           std::is_same<T,uint64_t>::value||
           std::is_same<T,float_t>::value||
           std::is_same<T,double_t>::value)
        {

        }
        else
        {
            throw Exception(1, "Type not support .", __FILE__,__LINE__);
        }

        for (typename std::vector<T>::iterator it = v.begin(); it != v.end(); it++)
        {
            if(needASCII)
                std::cout << static_cast<int>(*it) << std::endl;
            else
                std::cout << *it << std::endl;
        }
    }

    template<typename T>
    static void saveVector(std::vector<T>& v,const char* path,const char* format)
    {
        if(std::is_same<T,char>::value||
           std::is_same<T,unsigned char>::value||
           std::is_same<T,int>::value||
           std::is_same<T,unsigned int>::value||
           std::is_same<T,float>::value||
           std::is_same<T,double>::value||
           std::is_same<T,long>::value||
           std::is_same<T,unsigned long>::value||
           std::is_same<T,long long>::value||
           std::is_same<T,unsigned long long>::value||
           std::is_same<T,short>::value||
           std::is_same<T,unsigned short>::value||
           std::is_same<T,int8_t>::value||
           std::is_same<T,uint8_t>::value||
           std::is_same<T,int16_t>::value||
           std::is_same<T,uint16_t>::value||
           std::is_same<T,int32_t>::value||
           std::is_same<T,uint32_t>::value||
           std::is_same<T,int64_t>::value||
           std::is_same<T,uint64_t>::value||
           std::is_same<T,float_t>::value||
           std::is_same<T,double_t>::value)
        {

        }
        else
        {
            throw Exception(1, "Type not support .", __FILE__,__LINE__);
        }
        std::ofstream outfile(path,std::ios::trunc);
        if(!outfile)
        {
            throw Exception(1, "File open err.", __FILE__,__LINE__);
        }
        for (typename std::vector<T>::iterator it = v.begin(); it != v.end(); it++)
        {
            if(std::is_same<T,char>::value||std::is_same<T,int8_t>::value)
            {
                outfile<<static_cast<int>(*it)<<format;
            }
            else if(std::is_same<T,unsigned char>::value||std::is_same<T,uint8_t>::value)
            {
                outfile<<static_cast<unsigned int>(*it)<<format;
            }
            else
            {
                outfile<<*it<<format;
            }

        }
        outfile.flush();
        outfile.close();
    }

    template<typename T>
    static void readVector(std::vector<T>& v,const char* path,const char* format)
    {
        if(std::is_same<T,char>::value||
           std::is_same<T,unsigned char>::value||
           std::is_same<T,int>::value||
           std::is_same<T,unsigned int>::value||
           std::is_same<T,float>::value||
           std::is_same<T,double>::value||
           std::is_same<T,long>::value||
           std::is_same<T,unsigned long>::value||
           std::is_same<T,long long>::value||
           std::is_same<T,unsigned long long>::value||
           std::is_same<T,short>::value||
           std::is_same<T,unsigned short>::value||
           std::is_same<T,int8_t>::value||
           std::is_same<T,uint8_t>::value||
           std::is_same<T,int16_t>::value||
           std::is_same<T,uint16_t>::value||
           std::is_same<T,int32_t>::value||
           std::is_same<T,uint32_t>::value||
           std::is_same<T,int64_t>::value||
           std::is_same<T,uint64_t>::value||
           std::is_same<T,float_t>::value||
           std::is_same<T,double_t>::value)
        {

        }
        else
        {
            throw Exception(1, "Type not support .", __FILE__,__LINE__);
        }
        std::ifstream inFile(path,std::ios::in);
        if(!inFile)
        {
            throw Exception(1, "File open err.", __FILE__,__LINE__);
        }
        std::ostringstream tmpStr;
        tmpStr << inFile.rdbuf();
        std::string str = tmpStr.str();
        inFile.close();
        std::vector<std::string> datas;
        ExString::split(datas,str,format);

        for (auto it = datas.begin(); it != datas.end(); it++)
        {
            std::istringstream iss(*it);

            if(std::is_same<T,char>::value||std::is_same<T,int8_t>::value)
            {
                int num;
                iss>>num;
                v.push_back(num);
            }
            else if(std::is_same<T,unsigned char>::value||std::is_same<T,uint8_t>::value)
            {
                unsigned int num;
                iss>>num;
                v.push_back(num);
            }
            else
            {
                T num;
                iss>>num;
                v.push_back(num);
            }
        }
    }

    static void readVectorStr(std::vector<std::string>& v,const char* path,const char* format)
    {
        std::ifstream inFile(path,std::ios::in);
        if(!inFile)
        {
            throw Exception(1, "File open err.", __FILE__,__LINE__);
        }
        std::ostringstream tmpStr;
        tmpStr << inFile.rdbuf();
        std::string str = tmpStr.str();
        inFile.close();
        ExString::split(v,str,format);
    }

};
}

#endif 

