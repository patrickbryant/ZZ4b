// -*- C++ -*-

#if !defined(initBranch_H)
#define initBranch_H

#include <iostream>
#include <TChain.h>
#include <TChainElement.h>
#include <TDataType.h>
#include <TClass.h>

//template <typename T> std::string type_name();

namespace NtupleAna {

  static inline void initBranch(TChain *tree, std::string name, auto& variable){
    const char *bname = name.c_str();
    tree->SetBranchStatus(bname, 1);

    //TChainElement* element = (TChainElement*) tree->FindObject(bname);
    //element->SetBaddress(add);
    //element->SetBranchPtr(ptr);
    //int res = tree->CheckBranchAddressType(branch, TClass::GetClass(element->GetBaddressClassName()), (EDataType) element->GetBaddressType(), element->GetBaddressIsPtr());
    //int code = tree->SetBranchAddress(bname, add, ptr, TClass::GetClass(element->GetBaddressClassName()), (EDataType) element->GetBaddressType(), element->GetBaddressIsPtr());

    //TBranch* branch = tree->GetBranch(bname);
    //TBranch** ptr = &branch;
    //TClass* expectedClass;
    //EDataType expectedType;
    //branch->GetExpectedType(expectedClass, expectedType);
    //std::cout << "initBranch::" << bname << " " << expectedType << " " << typeid(variable).name() << std::endl;
    
    int code = tree->SetBranchAddress(bname, &variable);
    if(code != 0) std::cout << "initBranch::WARNING " << bname << " " << code << std::endl;
  }

  static inline void initBranch(TChain *tree, std::string name, void* add){
    const char *bname = name.c_str();
    tree->SetBranchStatus(bname, 1);
    int code = tree->SetBranchAddress(bname, add);
    if(code != 0) std::cout << "initBranch::WARNING " << bname << " " << code << std::endl;
  }

}
#endif // initBranch_H
