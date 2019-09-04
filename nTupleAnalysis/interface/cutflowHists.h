// -*- C++ -*-
#if !defined(cutflowHists_H)
#define cutflowHists_H

#include <iostream>
#include <TH1F.h>
#include <TH2F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"

namespace nTupleAnalysis {

  class cutflowHists {
  public:
    TFileDirectory dir;
    
    TH1F* unitWeight;
    TH1F* weighted;

    TH2F* truthM4b = NULL;

    cutflowHists(std::string, fwlite::TFileService&, bool);
    void BasicFill(std::string, eventData*);
    void Fill(std::string, eventData*);
    void labelsDeflate();
    ~cutflowHists(); 

  };

}
#endif // cutflowHists_H
