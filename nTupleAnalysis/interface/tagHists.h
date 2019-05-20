// -*- C++ -*-
#if !defined(tagHists_H)
#define tagHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/nTupleAnalysis/interface/eventData.h"
#include "ZZ4b/nTupleAnalysis/interface/eventView.h"
#include "ZZ4b/nTupleAnalysis/interface/eventHists.h"

namespace nTupleAnalysis {

  class tagHists {
  public:
    TFileDirectory dir;
    
    // Object Level
    eventHists* threeTag;
    eventHists* fourTag;

    tagHists(std::string, fwlite::TFileService&, bool doViews = false, bool isMC = false, bool blind = true);
    void Fill(eventData*);
    void Fill(eventData* event, std::vector<std::unique_ptr<eventView>> &views);
    ~tagHists(); 

  };

}
#endif // tagHists_H
