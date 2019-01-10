// -*- C++ -*-

#if !defined(eventView_H)
#define eventView_H
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/dijet.h"

namespace NtupleAna {
  //eventView object
  class eventView {

  public:
    dijet lead;
    dijet subl;

    TLorentzVector* p;
    float pt;
    float eta;
    float phi;
    float m;
    float e;

    eventView(dijet, dijet); 
    ~eventView(); 

    //void dump();
  };

}
#endif // eventView_H

