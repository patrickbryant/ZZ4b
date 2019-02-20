// -*- C++ -*-

#if !defined(jetData_H)
#define jetData_H
#include <TChain.h>
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/initBranch.h"

namespace NtupleAna {
  //forward declaration for use in jet constructor from tree
  class jetData;
  //jet object
  class jet {

  public:
    float pt;
    float eta;
    float phi;
    float m;
    float e;
    TLorentzVector p;

    float deepCSV;
    float CSVv2;

    jet();
    jet(UInt_t, jetData*); 
    jet(TLorentzVector&, float tag = -1); 
    ~jet(); 

    //void dump();
  };

  //class for tree access
  class jetData {

  public:
    UInt_t n;

    float pt [40];
    float eta[40];
    float phi[40];
    float m  [40];

    float deepCSV[40];
    float CSVv2  [40];

    jetData(std::string, TChain*); 
    std::vector<jet*> getJets(float ptMin = -1e6, float etaMax = 1e6, float tagMin = -1e6, std::string tagger = "CSVv2");
    ~jetData(); 

    //void dump();
  };

}
#endif // jetData_H

