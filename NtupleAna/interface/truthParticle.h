// -*- C++ -*-

#if !defined(truthParticle_H)
#define truthParticle_H
#include <TChain.h>
#include <TLorentzVector.h>
#include "ZZ4b/NtupleAna/interface/initBranch.h"

namespace NtupleAna {
  //forward declaration for use in particle constructor from tree
  class truthParticle;
  //particle object
  class particle {

  public:
    float pt;
    float eta;
    float phi;
    float m;
    float e;
    TLorentzVector p;

    Int_t genPartIdxMother;
    Int_t pdgId;

    particle();
    particle(UInt_t, truthParticle*); 
    ~particle(); 

    //void dump();
  };

  //class for tree access
  class truthParticle {

  public:
    UInt_t n;

    float pt [300];
    float eta[300];
    float phi[300];
    float m  [300];

    Int_t genPartIdxMother[300];
    Int_t pdgId[300];

    truthParticle(std::string, TChain*); 
    std::vector<std::shared_ptr<particle>> getParticles(Int_t absPDG = -1, Int_t absMomPDG = -1);
    ~truthParticle(); 

    //void dump();
  };

}
#endif // truthParticle_H

