// -*- C++ -*-
#if !defined(brilCSV_H)
#define brilCSV_H

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include "DataFormats/FWLite/interface/InputSource.h" //for edm::LuminosityBlockRange
 
/*
 * A class to read data from a csv file. Based on https://thispointer.com/how-to-read-data-from-a-csv-file-in-c/
 */
class brilCSV
{
  std::string fileName;
 
 public:
 brilCSV(std::string filename) :
  fileName(filename)
  { }
 
  // Function to fetch data from a CSV File
  std::map<edm::LuminosityBlockID, float> GetData();
};
 
#endif // brilCSV_H

