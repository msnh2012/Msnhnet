#include "Msnhnet/3rdparty/yaml-cpp/null.h"

namespace YAML {
_Null Null;

bool IsNullString(const std::string& str) {
  return str.empty() || str == "~" || str == "null" || str == "Null" ||
         str == "NULL";
}
}
