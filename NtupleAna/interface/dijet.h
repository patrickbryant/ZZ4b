// -*- C++ -*-

#if !defined(dijet_H)
#define dijet_H
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/jetData.h"

namespace NtupleAna {
  //dijet object
  class dijet {

  public:
    jet lead;
    jet subl;

    TLorentzVector p;
    float pt;
    float eta;
    float phi;
    float m;
    float e;

    dijet();
    dijet(jet&, jet&); 
    ~dijet(); 

    //void dump();
  };

}
#endif // dijet_H

