// -*- C++ -*-
#if !defined(viewHists_H)
#define viewHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/eventView.h"
#include "ZZ4b/NtupleAna/interface/vecHists.h"

namespace NtupleAna {

  class viewHists {
  public:
    TFileDirectory dir;
    
    // Object Level

    // Event Level
    TH1F* dHH;
    TH1F* xZZ;
    TH1F* mZZ;
    TH1F* mZH;

    viewHists(std::string, fwlite::TFileService&);
    void Fill(eventData*, eventView&);
    ~viewHists(); 

  };

}
#endif // viewHists_H
