#include "ZZ4b/NtupleAna/interface/brilCSV.h"

/*
 * Parses through csv file line by line and returns the data
 * in vector of vector of strings.
 */
std::map<edm::LuminosityBlockID, float> brilCSV::GetData()
{
  std::ifstream file(fileName);
 
  std::map<edm::LuminosityBlockID, float> lumiData;
 
  std::string line = "";
  float intLumi = 0;
  // Iterate through each line and split the content using delimeter
  while (getline(file, line))
    {
      if(line.substr(0,1)=="#") continue; //comment, skip this line

      //line format is run:fill,lumiBlock:lumiBlock,date hh:mm:ss,hltpath,delivered(/pb),recorded(/pb),avgpu,source
      std::vector<std::string> vecLine;
      boost::algorithm::split(vecLine, line, boost::is_any_of(",:"));

      unsigned int run;
      unsigned int lumiBlock;
      run       = std::stoul(vecLine[0]);
      lumiBlock = std::stoul(vecLine[2]);
      edm::LuminosityBlockID lumiID(run, lumiBlock);
      float lumi;
      lumi      = std::stof (vecLine[9]);
      intLumi+=lumi;
      lumiData[lumiID] = lumi;
      //std::cout << run << " " << lumiBlock << " " << lumi << " " << intLumi << std::endl;
    }
  // Close the File
  file.close();
 
  return lumiData;
}
