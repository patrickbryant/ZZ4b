// -*- C++ -*-
#if !defined(tagHists_H)
#define tagHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/eventView.h"
#include "ZZ4b/NtupleAna/interface/eventHists.h"

namespace NtupleAna {

  class tagHists {
  public:
    TFileDirectory dir;
    
    // Object Level
    eventHists* threeTag;
    eventHists* fourTag;

    tagHists(std::string, fwlite::TFileService&, bool doViews = false, bool blind = true);
    void Fill(eventData*);
    void Fill(eventData* event, std::vector<eventView*> &views);
    ~tagHists(); 

  };

}
#endif // tagHists_H
