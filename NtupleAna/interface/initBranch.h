// -*- C++ -*-

#if !defined(initBranch_H)
#define initBranch_H

#include <TTree.h>

namespace NtupleAna {

  static void initBranch(TTree *tree, std::string name, void *add){
    const char *bname = name.c_str();
    tree->SetBranchStatus(bname, 1);
    tree->SetBranchAddress(bname, add);
  }

}
#endif // initBranch_H
