// -*- C++ -*-

#if !defined(dijet_H)
#define dijet_H
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/jetData.h"

namespace NtupleAna {
  const float mH = 125.0;
  const float mZ =  91.0;

  //dijet object
  class dijet {

  public:
    jet lead;
    jet subl;

    TLorentzVector p;
    float dR;
    float st;
    float pt;
    float eta;
    float phi;
    float m;
    float e;

    TLorentzVector pZ;
    TLorentzVector pH;

    dijet();
    dijet(jet&, jet&); 
    ~dijet(); 

    //void dump();
  };

}
#endif // dijet_H

