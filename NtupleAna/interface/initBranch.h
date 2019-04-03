// -*- C++ -*-

#if !defined(initBranch_H)
#define initBranch_H

#include <iostream>
#include <TTree.h>

namespace NtupleAna {

  static inline void initBranch(TTree *tree, std::string name, void *add){
    const char *bname = name.c_str();
    int status = tree->SetBranchAddress(bname, add);
    if(status != 0) std::cout << "initBranch::WARNING " << bname << " " << status << std::endl;
  }

}
#endif // initBranch_H
