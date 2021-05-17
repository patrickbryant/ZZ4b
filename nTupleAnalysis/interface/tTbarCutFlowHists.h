// -*- C++ -*-
#if !defined(tTbarCutFlowHists_H)
#define tTbarCutFlowHists_H

#include <iostream>
#include <TH1F.h>
#include <TH2F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/tTbarEventData.h"

namespace nTupleAnalysis {

  class tTbarCutFlowHists {
  public:
    bool debug = false;
    TFileDirectory dir;
    
    TH1I* unitWeight;
    TH1D* weighted;

    tTbarCutFlowHists(std::string, fwlite::TFileService&, bool, bool);
    void BasicFill(const std::string&, tTbarEventData*);
    void BasicFill(const std::string&, tTbarEventData*, float weight);
    void Fill(const std::string&, tTbarEventData*);

    void labelsDeflate();

    void AddCut(std::string cut);
    
    ~tTbarCutFlowHists(); 

  };

}
#endif // tTbarCutFlowHists_H
