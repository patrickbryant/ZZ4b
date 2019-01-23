// -*- C++ -*-

#if !defined(muonData_H)
#define muonData_H
#include <TChain.h>
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
    TLorentzVector p;

    int softId;
    int highPtId;

    int mediumId;
    int tightId;

    int quality;

    int jetIdx;
    float isolation;
    float dR = 1e6;

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
    int highPtId[10];

    int mediumId[10];
    int tightId[10];

    int jetIdx[10];
    float pfRelIso04_all[10];
    
    muonData(std::string, TChain*); 
    std::vector<muon*> getMuons(float ptMin = -1e6, float etaMax = 1e6, int tag = -1, bool isolation = false);
    ~muonData(); 

    //void dump();
  };

}
#endif // muonData_H

