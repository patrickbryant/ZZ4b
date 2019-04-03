// -*- C++ -*-

#if !defined(trigData_H)
#define trigData_H
#include <TChain.h>
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/initBranch.h"

namespace NtupleAna {
  //forward declaration for use in trig constructor from tree
  class trigData;
  //trig object
  class trig {

  public:
    float pt;
    float eta;
    float phi;
    float m;
    float e;
    TLorentzVector p;

    float l1pt;
    float l1pt_2;
    float l2pt;
    int id;
    int filterBits;


    trig();
    trig(UInt_t, trigData*); 
    ~trig(); 

    //void dump();
  };

  //class for tree access
  class trigData {

  public:
    UInt_t n;

    float pt [40];
    float eta[40];
    float phi[40];

    float l1pt[40];
    float l1pt_2[40];
    float l2pt[40];
    int id[40];
    int filterBits[40];

    trigData(std::string, TChain*); 
    std::vector<std::shared_ptr<trig>> getTrigs(float ptMin = -1e6, float etaMax = 1e6, int objId = 0);
    ~trigData(); 

    //void dump();
  };

}
#endif // trigData_H

