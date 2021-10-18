#if !defined(utils_H)
#define utils_H

#include <vector>
#include <map>
#include <string>

namespace nTupleAnalysis{
  namespace utils{
    std::vector<std::string> splitString (const std::string &str, const std::string &delim);
    std::string fillString(std::string str, const std::map<std::string, std::string> &keywords);
  }
}
#endif // utils_H