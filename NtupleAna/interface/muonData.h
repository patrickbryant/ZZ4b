// -*- C++ -*-

#if !defined(muonData_H)
#define muonData_H
#include <TLorentzVector.h>

namespace NtupleAna {
  //forward declaration for use in muon constructor from tree
  class muonData;
  //muon object
  class muon {

  public:
    float pt;
    float eta;
    float phi;
    float m;
    TLorentzVector* p;

    int softId;
    int mediumId;
    int tightId;

    muon(UInt_t, muonData*); 
    ~muon(); 

    //void dump();
  };

  //class for tree access
  class muonData {

  public:
    UInt_t n;

    float pt [10];
    float eta[10];
    float phi[10];
    float m  [10];

    float softId[10];
    float mediumId[10];
    float tightId[10];

    muonData(std::string, TChain*); 
    std::vector<muon> getMuons(float ptMin = -1e6, float etaMax = 1e6, float tagMin = -1e6);
    ~muonData(); 

    //void dump();
  };

}
#endif // muonData_H

