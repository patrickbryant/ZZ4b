#include "ZZ4b/nTupleAnalysis/interface/utils.h"
#include <regex>

namespace nTupleAnalysis{
  namespace utils{

    std::vector<std::string> splitString (const std::string &str, const std::string &delim){
      std::vector<std::string> strs;
      size_t start = 0;
      size_t end = 0;
      while ((end = str.find(delim, start)) != std::string::npos){
        strs.push_back(str.substr(start, end - start));
        start = end + delim.length();
      }
      strs.push_back(str.substr(start, str.length()));
      return strs;
    }

    std::string fillString(std::string str, const std::map<std::string, std::string> &keywords){
      for(const auto& [key, value] : keywords){
        std::regex keyword( "\\/\\*" + key + "\\*\\/");
        str = std::regex_replace(str, keyword, value);
      }
      return str;
    }
  }
}
