// -*- C++ -*-
#if !defined(eventHists_H)
#define eventHists_H

#include <iostream>
#include <TH1F.h>
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "ZZ4b/NtupleAna/interface/eventData.h"
#include "ZZ4b/NtupleAna/interface/jetHists.h"

namespace NtupleAna {

  class eventHists {
  public:
    TFileDirectory dir;
    
    
    TH1F*     nAllJets;
    jetHists*  allJets;
    TH1F*     nSelJets;
    jetHists*  selJets;
    TH1F*     nTagJets;
    jetHists*  tagJets;

    TH1F* nAllMuons;
    TH1F* nIsoMuons;

    eventHists(std::string, fwlite::TFileService&);
    void Fill(eventData*);
    ~eventHists(); 

  };

}
#endif // eventHists_H
