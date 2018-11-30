// -*- C++ -*-

#if !defined(jetData_H)
#define jetData_H
#include <TLorentzVector.h>

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
    TLorentzVector* p;

    float deepCSV;

    jet(UInt_t, jetData*); 
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

    jetData(std::string, TChain*); 
    std::vector<jet> getJets(float ptMin = -1e6, float etaMax = 1e6, float tagMin = -1e6);
    ~jetData(); 

    //void dump();
  };

}
#endif // jetData_H

