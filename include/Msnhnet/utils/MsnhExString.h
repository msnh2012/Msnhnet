#ifndef MSNHEXSTRING_H
#define MSNHEXSTRING_H
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <regex>
#include "Msnhnet/utils/MsnhException.h"
#include "Msnhnet/utils/MsnhExport.h"

#ifndef WIN32
#include <string.h>
#endif

namespace Msnhnet
{
class MsnhNet_API ExString
{
public:

    static void split(std::vector<std::string> &result, const std::string& str, const std::string& delimiter);

    static void toUpper(std::string &s);

    static bool isUpper(std::string &s);

    static bool isLower(std::string &s);

    static void toLower(std::string &s);

    static void leftTrim(std::string &s);

    static void rightTrim(std::string &s);

    static void trim(std::string &s);

    static void deleteMark(std::string& s, const std::string& mark);

    static bool isEmpty(const std::string &s);

    static bool isNum(const std::string &s);

    static bool isEqual(const std::string &a, const std::string &b);
    static bool isEqual(const char *a, const char *b);

    static bool strToInt(const std::string &s,int &i);

    static bool strToFloat(const std::string &s, float &i);

    static bool strToDouble(const std::string &s, double &i);

    static bool isEmail(const std::string& str);

    static bool isChinese(const std::string& str);

    static bool isNumAndChar(const std::string& str);

    static bool isChar(const std::string& str);

    static bool isHex(const std::string& str);

    static std::string left(const std::string& str, const int& n);

    static std::string right(const std::string& str, const int& n);

    static std::string mid(const std::string& str, const size_t &start, const size_t &offset);

    static std::vector<std::string> splitHex(const std::string& str);
};
}

#endif 
