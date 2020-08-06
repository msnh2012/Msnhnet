#include "Msnhnet/utils/MsnhExString.h"

namespace Msnhnet
{
void ExString::split(std::vector<std::string>& result,const std::string &str, const std::string &delimiter)
{
    char* save = nullptr;
#ifdef WIN32
    char* token = strtok_s(const_cast<char*>(str.c_str()), delimiter.c_str(), &save);
#else
    char* token = strtok_r(const_cast<char*>(str.c_str()), delimiter.c_str(), &save);
#endif
    while (token != nullptr)
    {
        result.emplace_back(token);
#ifdef WIN32
        token = strtok_s(nullptr, delimiter.c_str(), &save);
#else
        token = strtok_r(nullptr, delimiter.c_str(), &save);
#endif
    }
}

void ExString::toUpper(std::string& s)
{
    std::transform( std::begin(s),
                    std::end(s),
                    std::begin(s),
                    ::toupper
                    );
}

bool ExString::isUpper(std::string &s)
{
    return std::all_of( std::begin(s), std::end(s),
                        [] (char c) { return ::isupper(c); });
}

bool ExString::isLower(std::string &s)
{
    return std::all_of( std::begin(s), std::end(s),
                        [] (char c) { return ::islower(c); });
}

void ExString::toLower(std::string &s)
{
    std::transform( std::begin(s),
                    std::end(s),
                    std::begin(s),
                    ::tolower
                    );
}

void ExString::leftTrim(std::string& s)
{
    std::string whitespaces (" \t\f\v\n\r");
    size_t pos = s.find_first_not_of(whitespaces);
    if( pos != std::string::npos ) {
        s.erase(0, pos);
    } else {
        s.clear();
    }
}

void ExString::rightTrim(std::string &s)
{
    std::string whitespaces (" \t\f\v\n\r");
    size_t pos = s.find_last_not_of(whitespaces);
    if( pos != std::string::npos ) {
        s.erase(pos + 1);
    } else {
        s.clear();
    }
}

void ExString::trim(std::string &s)
{
    deleteMark(s,"\t");
    deleteMark(s,"\r");
    deleteMark(s,"\n");
    deleteMark(s," ");
}

void ExString::deleteMark(std::string &s, const std::string& mark)
{
    size_t nSize = mark.size();
    while(1)
    {
        size_t pos = s.find(mark);
        if(pos == std::string::npos)
        {
            return ;
        }

        s.erase(pos, nSize);
    }

}

bool ExString::isEmpty(const std::string &s)
{
    for (size_t i = 0; i < s.size(); i ++) {
        if (s[i] != ' ') return false;
    }

    return true;
}

bool ExString::isEqual(const std::string &a, const std::string &b)
{
    size_t sz = a.length();
    if (b.length() != sz)
        return false;
    for (size_t i = 0; i < sz; ++i)
        if (tolower(a[i]) != tolower(b[i]))
            return false;
    return true;
}

bool ExString::isEqual(const char *a, const char *b)
{
    bool eq = (strcmp(a, b) == 0);
    return eq;
}

bool ExString::strToInt(const std::string &s, int &i)
{
    std::string temp = s;

    if(!isNum(temp))
        return false;
    i=atoi(s.data());

    return true;
}

bool ExString::strToFloat(const std::string &s, float &i)
{
    if(!isNum(s))
        return false;
    i=(float)atof(s.data());

    return true;
}

bool ExString::strToDouble(const std::string &s, double &i)
{
    if(!isNum(s))
        return false;
    i=atof(s.data());
    return true;
}

bool ExString::isEmail(const std::string &str)
{
    const std::regex pattern("(\\w+)(\\.|_)?(\\w*)@(\\w+)(\\.(\\w+))+");
    return std::regex_match(str, pattern);
}

bool ExString::isChinese(const std::string &str)
{
    const std::regex pattern("^[\u4e00-\u9fa5]{0,}$");
    return std::regex_match(str, pattern);
}

bool ExString::isNumAndChar(const std::string &str)
{
    const std::regex pattern("^[A-Za-z0-9]+$");
    return std::regex_match(str, pattern);
}

bool ExString::isChar(const std::string &str)
{
    const std::regex pattern("^[A-Fa-f]+$");
    return std::regex_match(str, pattern);
}

bool ExString::isHex(const std::string &str)
{
    const std::regex pattern("^[A-Fa-f0-9]+$");
    return std::regex_match(str, pattern);
}

std::string ExString::left(const std::string &str, const int &n)
{
    if(n < 0)
    {
        throw Exception(0,"n must > 0",__FILE__,__LINE__, __FUNCTION__);
    }

    if(n > static_cast<int>(str.length()))
    {
        throw Exception(0,"n must < length of str",__FILE__,__LINE__, __FUNCTION__);
    }

    return str.substr(0,n);
}

std::string ExString::right(const std::string &str, const int &n)
{
    if(n<0)
    {
        throw Exception(0,"n must > 0",__FILE__,__LINE__, __FUNCTION__);
    }

    if(n>static_cast<int>(str.length()))
    {
        throw Exception(0,"n must < length of str",__FILE__,__LINE__, __FUNCTION__);
    }

    return str.substr(str.length()-n,n);
}

std::string ExString::mid(const std::string &str, const size_t &start, const size_t &offset)
{
    if(start<0)
    {
        throw Exception(0,"start must > 0",__FILE__,__LINE__, __FUNCTION__);
    }
    else if(start > static_cast<int>(str.length()))
    {
        throw Exception(0,"start must < length of str",__FILE__,__LINE__, __FUNCTION__);
    }

    if(offset<0)
    {
        throw Exception(0,"offset must > 0",__FILE__,__LINE__, __FUNCTION__);
    }
    else if(offset > static_cast<int>(str.length()))
    {
        throw Exception(0,"start must < length of str",__FILE__,__LINE__, __FUNCTION__);
    }

    if((start + offset) >static_cast<int>(str.length()))
    {
        throw Exception(0,"start + offset must < length of str",__FILE__,__LINE__, __FUNCTION__);
    }

    return str.substr(start, offset);
}

std::vector<std::string> ExString::splitHex(const std::string &str)
{
    if((str.length()%2)!=0)
    {
        throw Exception(0,"String length must be even",__FILE__,__LINE__, __FUNCTION__);
    }
    std::vector<std::string> temp;
    for(size_t i =0;i<str.length()/2;i++)
    {
        temp.push_back(str.substr(i*2,2));
    }

    return temp;
}

bool ExString::isNum(const std::string &s)
{
    const std::regex pattern("-?[0-9]*.?[0-9]*");
    return std::regex_match(s, pattern);
}
}
