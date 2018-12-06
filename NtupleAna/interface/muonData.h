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

    int jetIdx;

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

    int softId[10];
    int mediumId[10];
    int tightId[10];

    int jetIdx[10];
    
    muonData(std::string, TChain*); 
    std::vector<muon> getMuons(float ptMin = -1e6, float etaMax = 1e6, int tagMin = -1, bool isolation = false);
    ~muonData(); 

    //void dump();
  };

}
#endif // muonData_H

