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
    UChar_t cleanmask;

    float pt;
    float eta;
    float phi;
    float m;
    float e;
    TLorentzVector p;

    float bRegCorr;

    float deepB;
    float CSVv2;
    float deepFlavB;

    jet();
    jet(UInt_t, jetData*); 
    jet(TLorentzVector&, float tag = -1); 
    void bRegression();
    ~jet(); 

    //void dump();
  };

  //class for tree access
  class jetData {

  public:
    UInt_t n;
    //ULong64_t n;

    UChar_t cleanmask[40];

    float pt [40];
    float eta[40];
    float phi[40];
    float m  [40];

    float bRegCorr[40];

    float deepB[40];
    float CSVv2[40];
    float deepFlavB[40];

    jetData(std::string, TChain*); 
    std::vector< std::shared_ptr<jet> > getJets(float ptMin = -1e6, float ptMax = 1e6, float etaMax = 1e6, bool clean = false, float tagMin = -1e6, std::string tagger = "CSVv2", bool antiTag = false);
    std::vector< std::shared_ptr<jet> > getJets(std::vector< std::shared_ptr<jet> > inputJets, 
						float ptMin = -1e6, float ptMax = 1e6, float etaMax = 1e6, bool clean = false, float tagMin = -1e6, std::string tagger = "CSVv2", bool antiTag = false);
    ~jetData(); 

    //void dump();
  };


}
#endif // jetData_H

